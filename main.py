# ---------------- CONFIGURAÇÕES ----------------
import pandas as pd
from pandas import (DataFrame,Series,concat,date_range,read_sql,to_datetime,to_numeric,Timestamp,DateOffset,NA,MultiIndex)
from pandas.tseries.offsets import MonthEnd
import numpy as np
from numpy import (floor as np_floor,inf as np_inf,isinf as np_isinf,where as np_where)
from scipy.optimize import newton, brentq
from datetime import date, datetime, timedelta
import logging
from fastapi.responses import JSONResponse
import os, re, traceback
from functools import lru_cache
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # trocar para domínio do frontend em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(levelname)s %(message)s")

logger = logging.getLogger("price_manager")

SLEEP_BETWEEN_TICKERS = float(os.getenv("SLEEP_BETWEEN_TICKERS", "0.5"))  # para evitar throttling

# ---------------- FUNÇÕES AUXILIARES ----------------
def limpar_valor(col):
    return pd.to_numeric(
        col.astype(str)
        .str.replace("R$", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace([' - ', '-', '', 'nan'], pd.NA),
        errors='coerce'
    )

def raiz_ticker(ticker):
    return str(ticker).strip()[:4].upper()

def normalizar_ticker(ticker):
    return re.sub(r"F$", "", str(ticker).strip())

# ---------------- ETAPAS DO PIPELINE ----------------
def carregar_dados():
    df_mov = pd.read_sql("SELECT * FROM ldinvest_movimentacao_07122025", con=engine)
    df_neg = pd.read_sql("SELECT * FROM ldinvest_negociacao_07122025", con=engine)
    cnpj_b3 = pd.read_sql("SELECT * FROM cnpj_b3", con=engine)
    return df_mov, df_neg, cnpj_b3

def preparar_dados(df_mov, df_neg):
    for col in ['Preço', 'Valor', 'Quantidade']:
        df_neg[col] = limpar_valor(df_neg[col])
    for col in ['Preço unitário', 'Valor da Operação', 'Quantidade']:
        df_mov[col] = limpar_valor(df_mov[col])

    df_mov['Ticker'] = df_mov['Produto'].str.extract(r'^([A-Z0-9]+)')
    df_neg['Ticker'] = df_neg['Código de Negociação'].apply(normalizar_ticker)

    df_neg['Data do Negócio'] = pd.to_datetime(df_neg['Data do Negócio'], dayfirst=True, errors='coerce')
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], dayfirst=True, errors='coerce')
    return df_mov, df_neg

def processar_eventos(df_mov):
    # desdobros
    desdobros = df_mov[df_mov['Movimentação'] == 'Desdobro'].groupby(['Ticker'])['Quantidade'].sum().reset_index()

    # subscrições
    subscricoes_original = df_mov[df_mov['Movimentação'].isin(['Direitos de Subscrição - Exercido'])].copy()
    subscricoes_original['Ticker Raiz'] = subscricoes_original['Ticker'].apply(raiz_ticker)
    subscricoes_original['Total Investido'] = subscricoes_original['Quantidade'] * subscricoes_original['Preço unitário']
    subscricoes = subscricoes_original.groupby(['Ticker Raiz'], as_index=False)[['Quantidade', 'Total Investido']].sum()
    subscricoes.rename(columns={'Quantidade': 'Qtd_subscr'}, inplace=True)

    # bonificações
    bonus_original = df_mov[df_mov['Movimentação'] == 'Bonificação em Ativos'].copy()
    bonus_original['Quantidade'] = np.floor(bonus_original['Quantidade']).astype(int)
    bonus_original['Investido Bonus'] = bonus_original['Quantidade'] * bonus_original['Preço unitário']
    bonus = bonus_original.groupby('Ticker').agg({
        'Quantidade': 'sum',
        'Investido Bonus': 'sum'
    }).reset_index()

    # leilão de fração
    leilao_original = df_mov[df_mov['Movimentação'] == 'Leilão de Fração'].copy()
    leilao = leilao_original.groupby('Ticker')['Quantidade'].sum().reset_index()
    leilao.columns = ['Ticker', 'Qtd Leiloada']

    rendimentos = df_mov[df_mov['Movimentação'] == 'Rendimento'].copy()

    # Coluna auxiliar: identificar se o produto é FII
    filtros_fii = ['FII', 'FDO', 'FUNDO DE INVESTIMENTO IMOBILIARIO']
    rendimentos['É FII'] = rendimentos['Produto'].str.upper().str.contains('|'.join(filtros_fii), na=False)

    # Extrai o ticker
    rendimentos['Ticker'] = rendimentos['Produto'].str.extract(r'^([A-Z0-9]+)')

    rendimentos['Ticker Raiz'] = rendimentos['Ticker'].apply(raiz_ticker)

    # Agrupa separadamente
    rendimento_fii = rendimentos[rendimentos['É FII']].groupby('Ticker Raiz')['Valor da Operação'].sum().reset_index()
    rendimento_fii.columns = ['Ticker Raiz', 'Rendimento_fii']

    rendimento_acoes = rendimentos[~rendimentos['É FII']].groupby('Ticker Raiz')['Valor da Operação'].sum().reset_index()
    rendimento_acoes.columns = ['Ticker Raiz', 'Rendimento_acoes']

    return desdobros, subscricoes, subscricoes_original, bonus, bonus_original, leilao, leilao_original

def aplicar_grupamentos(df_neg):
    grups = {
        'IRBR3': {'fator': 30, 'data': datetime(2023, 1, 1)}
    }
    compras = df_neg[df_neg['Tipo de Movimentação'] == 'Compra'].copy()
    ajuste_grupamento = []

    for ticker, info in grups.items():
        fator, data_grupamento = info['fator'], info['data']
        comp_pre = compras[(compras['Ticker'] == ticker) & (compras['Data do Negócio'] < data_grupamento)].copy()
        comp_pos = compras[(compras['Ticker'] == ticker) & (compras['Data do Negócio'] >= data_grupamento)].copy()

        qtd_total = comp_pre['Quantidade'].sum() / fator + comp_pos['Quantidade'].sum()
        total_investido = (comp_pre['Quantidade'] * comp_pre['Preço']).sum() + (comp_pos['Quantidade'] * comp_pos['Preço']).sum()
        preco_medio = total_investido / qtd_total if qtd_total else 0

        ajuste_grupamento.append({
            'Ticker': ticker,
            'Qtd Final': qtd_total,
            'Total Investido': total_investido,
            'Preço Médio Ajustado': preco_medio
        })

    return ajuste_grupamento

def consolidar_carteira(df_mov, df_neg, desdobros, subscricoes, bonus, leilao, ajuste_grupamento):
    # compras
    compras = df_neg[df_neg['Tipo de Movimentação'] == 'Compra']
    compras = compras.groupby('Ticker').agg({
        'Quantidade': 'sum',
        'Preço': lambda x: (x * compras.loc[x.index, 'Quantidade']).sum()
    }).reset_index()
    compras.columns = ['Ticker', 'Qtd Compra', 'Total Compra']

    # vendas
    vendas = df_neg[df_neg['Tipo de Movimentação'] == 'Venda']

    vendas = vendas.groupby('Ticker').agg({
        'Quantidade': 'sum',
        'Preço': lambda x: (x * vendas.loc[x.index, 'Quantidade']).sum()
    }).reset_index()
    vendas.columns = ['Ticker', 'Qtd Vendida', 'Total Vendido']

    # consolidar
    df = compras.merge(vendas, on='Ticker', how='left')
    df['Ticker Raiz'] = df['Ticker'].apply(raiz_ticker)
    df = df.merge(leilao, on='Ticker', how='left')
    df = df.merge(bonus, on='Ticker', how='left')
    df = df.merge(subscricoes, on='Ticker Raiz', how='left')
    df = df.merge(desdobros, on='Ticker', how='left', suffixes=('', '_Desdobro'))

    for col in ['Qtd Vendida', 'Qtd Leiloada', 'Quantidade', 'Investido Bonus', 'Qtd_subscr', 'Total Investido', 'Quantidade_Desdobro']:
        df[col] = df[col].fillna(0)

    df['Qtd Final'] = df['Qtd Compra'] + df['Quantidade'] + df['Qtd_subscr'] + df['Quantidade_Desdobro'] - df['Qtd Vendida']
    df['Total Investido'] = df['Total Compra'] + df['Investido Bonus'] + df['Total Investido']

    # ajustes específicos (exemplo TIET4 → AESB3)
    tiet4_qtd = df.loc[df['Ticker'] == 'TIET4', 'Qtd Final'].sum() // 5
    df.loc[df['Ticker'] == 'AESB3', 'Qtd Final'] += tiet4_qtd
    df = df[df['Ticker'] != 'TIET4']
    
    qtd_total_historico = (
    df['Qtd Compra'].fillna(0) +
    df['Quantidade'].fillna(0) +
    df['Qtd_subscr'].fillna(0) +
    df['Quantidade_Desdobro'].fillna(0)
    )
    
    df['Preço Médio Ajustado'] = np.where(
    df['Qtd Final'] == 0,
    np.where(
        qtd_total_historico == 0,
        0,  # Se o investido for 0 e as Qtds também, o preço médio é 0
        df['Total Investido'] / qtd_total_historico
    ),
    df['Total Investido'] / df['Qtd Final']
    )

    df_carteira = df[['Ticker', 'Qtd Final', 'Total Investido', 'Preço Médio Ajustado','Qtd Vendida', 'Total Vendido']].sort_values('Ticker')

    # aplicar cisão AZEV4 → AZEV4 + AZTE3
    cisao = df_mov[df_mov['Movimentação'] == 'Cisão'].copy()
    cisao_azev = cisao[cisao['Produto'].str.contains("AZEV4|AZTE3", regex=True)]
    if not cisao_azev.empty:
        cisao_azev['Ticker'] = cisao_azev['Produto'].str.extract(r'^([A-Z0-9]+)')
        cisao_azev['Quantidade'] = pd.to_numeric(cisao_azev['Quantidade'], errors='coerce')
        qtd_azev = cisao_azev[cisao_azev['Ticker'] == 'AZEV4']['Quantidade'].sum()
        qtd_azte = cisao_azev[cisao_azev['Ticker'] == 'AZTE3']['Quantidade'].sum()
        custo_total = df_carteira.loc[df_carteira['Ticker'] == 'AZEV4', 'Total Investido'].values[0]
        custo_azev, custo_azte = custo_total * (2/3), custo_total * (1/3)
        df_carteira.loc[df_carteira['Ticker'] == 'AZEV4', 'Total Investido'] = custo_azev
        df_carteira.loc[df_carteira['Ticker'] == 'AZEV4', 'Preço Médio Ajustado'] = custo_azev / qtd_azev
        novo = pd.DataFrame([{'Ticker': 'AZTE3', 'Qtd Final': qtd_azte, 'Total Investido': custo_azte, 'Preço Médio Ajustado': custo_azte / qtd_azte,'Qtd Vendida':0.0,'Total Investido':0.0}])
        df_carteira = pd.concat([df_carteira, novo], ignore_index=True)

    # aplicar grupamentos
    df_carteira = df_carteira[~df_carteira['Ticker'].isin([a['Ticker'] for a in ajuste_grupamento])]
    df_carteira = pd.concat([df_carteira, pd.DataFrame(ajuste_grupamento)], ignore_index=True)
    # apenas ativos em carteira
    # df_carteira = df_carteira[df_carteira['Qtd Final'] > 0].copy()

    return(df_carteira)

def calcular_evolucao_temporal_jcp(df_mov):
    # 1. Garantir que a coluna Data é datetime
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], dayfirst=True)

    # 2. Filtrar apenas JCP
    proventos = df_mov[df_mov['Movimentação'].isin(['Juros Sobre Capital Próprio'])].copy()
    
    # 3. Agrupar e preparar a série temporal
    proventos = proventos.groupby(['Data', 'Movimentação'])['Valor da Operação'].sum().reset_index()
    proventos_ts = proventos.pivot_table(index='Data', columns='Movimentação', values='Valor da Operação', fill_value=0)
    
    if proventos_ts.empty:
        return {"ano_calendario_ir": 0.0, "total_acumulado": 0.0}

    proventos_ts['Total Proventos'] = proventos_ts.sum(axis=1)
    proventos_ts = proventos_ts.reset_index()

    # 4. Definição dos Períodos
    hoje = datetime.today()
    
    # Lógica para IR: 01/01 do ano passado até 31/12 do ano passado
    ano_anterior = hoje.year - 1
    inicio_ir = pd.Timestamp(year=ano_anterior, month=1, day=1)
    fim_ir = pd.Timestamp(year=ano_anterior, month=12, day=31)

    # 5. Filtros
    filtros = {
        "ano_calendario_ir": proventos_ts[(proventos_ts['Data'] >= inicio_ir) & (proventos_ts['Data'] <= fim_ir)],
        "mes_atual": proventos_ts[proventos_ts['Data'] >= hoje.replace(day=1)],
        "desde_inicio": proventos_ts
    }

    return {
        periodo: float(filtro['Total Proventos'].sum())
        for periodo, filtro in filtros.items()
    }

def calcular_evolucao_temporal_Dividendo(df_mov):
    # 1. Garantir que a coluna Data é datetime para evitar erros de comparação
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], dayfirst=True)

    # 2. Filtrar apenas Movimentações de 'Dividendo'
    proventos = df_mov[df_mov['Movimentação'].isin(['Dividendo'])].copy()
    
    # 3. Agrupar e preparar a série temporal
    proventos = proventos.groupby(['Data', 'Movimentação'])['Valor da Operação'].sum().reset_index()
    
    # Tratamento caso o DataFrame esteja vazio
    if proventos.empty:
        return {"ano_calendario_ir": 0.0, "total_acumulado": 0.0}
        
    proventos_ts = proventos.pivot_table(index='Data', columns='Movimentação', values='Valor da Operação', fill_value=0)
    proventos_ts['Total Proventos'] = proventos_ts.sum(axis=1)
    proventos_ts = proventos_ts.reset_index()

    # 4. Definição do Período de Imposto de Renda (Ano-Calendário)
    hoje = datetime.today()
    ano_anterior = hoje.year - 1
    
    inicio_ir = pd.Timestamp(year=ano_anterior, month=1, day=1)
    fim_ir = pd.Timestamp(year=ano_anterior, month=12, day=31)
    inicio_mes_atual = hoje.replace(day=1)

    # 5. Aplicação dos filtros temporais
    filtros = {
        "ano_calendario_ir": proventos_ts[(proventos_ts['Data'] >= inicio_ir) & (proventos_ts['Data'] <= fim_ir)],
        "mes_atual": proventos_ts[proventos_ts['Data'] >= inicio_mes_atual],
        "um_ano": proventos_ts[proventos_ts['Data'] >= hoje - pd.DateOffset(years=1)],
        "desde_inicio": proventos_ts
    }

    return {
        periodo: float(filtro['Total Proventos'].sum())
        for periodo, filtro in filtros.items()
    }

def gerar_json_ir(df, cnpj_b3, df_lucros):
    """
    Consolida os dados para o Relatório de IR.
    df_carteira_filtrada: Contém quantidade, preco_medio, total_investido, Dividendo, Juros Sobre Capital Próprio
    cnpj_b3: Tabela de de-para Ticker -> CNPJ e Razão Social
    df_lucros: Histórico de vendas com coluna 'lucro' e 'Data do Negócio'
    """
    
    hoje = datetime.today()
    ano_anterior = hoje.year - 1

    # --- 1. PREPARAÇÃO DA CARTEIRA (Bens e Direitos + Proventos) ---
    
    # Merge com a tabela de CNPJs para enriquecer os dados
    df_ir = pd.merge(df, cnpj_b3, on='Ticker', how='left')
    
    # Tratamento de valores nulos e renomeação para o padrão do Front
    df_ir['CNPJ'] = df_ir['CNPJ'].fillna("CNPJ não encontrado")
    df_ir['Razão Social'] = df_ir['Razão Social'].fillna("Razão Social não encontrada")
    
    # Selecionamos e renomeamos apenas as colunas que o front vai usar
    carteira_ir = df_ir.rename(columns={
        'Razão Social': 'Razao_Social'
    })[['Ticker', 'CNPJ', 'Razao_Social', 'quantidade', 'preco_medio', 'total_investido', 'dividendos', 'juros_sobre_capital_proprio']]
    
    # Converter para lista de dicionários
    lista_carteira = carteira_ir.to_dict(orient='records')

    # --- 2. PREPARAÇÃO DOS LUCROS MENSAIS (Renda Variável) ---
    
    # Garantir que a data está no formato correto e filtrar o ano anterior
    df_lucros['Data do Negócio'] = pd.to_datetime(df_lucros['Data do Negócio'])
    df_lucros_ir = df_lucros[df_lucros['Data do Negócio'].dt.year == ano_anterior].copy()
    
    # Criar um esqueleto dos 12 meses para garantir que o JSON tenha todos, mesmo com lucro zero
    meses_nomes = {
        1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
        7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
    }
    
    lucros_por_mes = df_lucros_ir.groupby(df_lucros_ir['Data do Negócio'].dt.month)['lucro'].sum().to_dict()
    
    lista_lucros_mensais = []
    for m in range(1, 13):
        lista_lucros_mensais.append({
            "mes": m,
            "mes_nome": meses_nomes[m],
            "lucro": float(lucros_por_mes.get(m, 0.0))
        })

    # --- 3. MONTAGEM DO JSON FINAL ---
    return {
        "ano_referencia": ano_anterior,
        "carteira_ir": lista_carteira,
        "lucros_mensais": lista_lucros_mensais
    }

def calcular_proventos_ir(df_mov):
    # 1. Preparação das datas
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], dayfirst=True)
    
    hoje = datetime.today()
    ano_anterior = hoje.year - 1
    inicio_ir = pd.Timestamp(year=ano_anterior, month=1, day=1)
    fim_ir = pd.Timestamp(year=ano_anterior, month=12, day=31)

    # 2. Filtro de Período (Ano-Calendário) e Tipos de Proventos
    tipos_proventos = ['Juros Sobre Capital Próprio', 'Dividendo', 'Rendimento']
    
    mask = (
        (df_mov['Data'] >= inicio_ir) & 
        (df_mov['Data'] <= fim_ir) & 
        (df_mov['Movimentação'].isin(tipos_proventos))
    )
    
    df_filtrado = df_mov[mask].copy()

    if df_filtrado.empty:
        return pd.DataFrame(columns=['Ticker', 'Dividendo', 'Juros Sobre Capital Próprio', 'Rendimento'])

    # 3. Extração do Ticker (Regex melhorada para pegar 4 letras e números)
    # Ex: "PETR4 - PETROLEO BRASILEIRO S.A." -> "PETR4"
    df_filtrado['Ticker'] = df_filtrado['Produto'].str.extract(r'^([A-Z0-9]{4,6})')

    # 4. Agrupamento e Pivotagem
    proventos_agrupados = (
        df_filtrado.groupby(['Ticker', 'Movimentação'])['Valor da Operação']
        .sum()
        .reset_index()
    )

    df_pivot = proventos_agrupados.pivot_table(
        index='Ticker', 
        columns='Movimentação', 
        values='Valor da Operação', 
        fill_value=0
    ).reset_index()

    # 5. Garantir que todas as colunas existam (mesmo que o usuário não tenha recebido um dos tipos)
    for col in tipos_proventos:
        if col not in df_pivot.columns:
            df_pivot[col] = 0.0

    return df_pivot

def extrair_raiz(ticker):
    # Expressão regular: captura letras no início da string
    match = re.match(r'([A-Za-z]+)', str(ticker))
    return match.group(1).upper() if match else None

def calcular_lucros_vendas(df_neg, df_mov, desdobros, subscricoes, bonus, leilao, ajuste_grupamento):


    vendas = df_neg[df_neg['Tipo de Movimentação'] == 'Venda']

    vendas = vendas.groupby(['Data do Negócio','Ticker']).agg({
        'Valor': 'sum',
        'Quantidade': 'sum',
    }).reset_index()


    # DATA_INICIO = vendas['Data do Negócio'].iloc[0]
    # DATA_FIM = vendas['Data do Negócio'].iloc[-1]
    
    # # Obter Tickers
    # TODOS_OS_TICKERS = vendas['Ticker'].unique().tolist()

    # Converte Data do Negócio para datetime para garantir que timedelta funcione
    vendas['Data do Negócio'] = pd.to_datetime(vendas['Data do Negócio'])

    # Gera o range de datas
    periodo_analise = vendas['Data do Negócio'].unique()

    # Lista para acumular resultados de cada mês
    historico_lucros = []
    # df_lucros_completo precisa ser definida antes do loop se você não está rodando em um ambiente iterativo
    # df_lucros_completo = pd.DataFrame(columns=["Data do Negócio", "Ticker", "lucro", "tipo venda"]) 

    df_lucros_completo = pd.DataFrame(columns=["Data do Negócio", "Ticker", "lucro", "tipo venda", "Preço Médio Ajustado"])

    # Loop principal
    for data_final_mes in periodo_analise:
        # if data_final_mes == periodo_analise[12]:
        #     break
        # --- 1. Calcular o dia anterior e Data de Reset PM ---
        
        # data_final_mes já é um objeto datetime (do pandas unique)
        dia_anterior_obj = data_final_mes - timedelta(days=1)
        dia_anterior_str = dia_anterior_obj.strftime('%Y-%m-%d')
        
        # 1a. Encontrar a última data de venda total para cada Ticker
        # Se df_lucros_completo não foi inicializada, esta linha pode falhar na primeira iteração.
        
        # Assumimos que está definida ou usamos uma inicialização segura.
        if 'df_lucros_completo' in locals() and not len(historico_lucros) == 0:
            # Filtra vendas totais passadas (antes ou na data anterior à atual)
            vendas_totais_historico = df_lucros_completo[
                (df_lucros_completo['tipo venda'] == 'venda total') & 
                (pd.to_datetime(df_lucros_completo['Data do Negócio']) <= dia_anterior_obj)
            ].copy()

            # Obtém a data máxima (última) de venda total para cada Ticker
            data_reset_pm = vendas_totais_historico.groupby('Ticker')['Data do Negócio'].max().reset_index()
            data_reset_pm.columns = ['Ticker', 'Data Reset PM']
            data_reset_pm['Data Reset PM'] = pd.to_datetime(data_reset_pm['Data Reset PM'])
        else:
            # Se for a primeira iteração, a base de reset está vazia
            data_reset_pm = pd.DataFrame({'Ticker': [], 'Data Reset PM': []})
        # --- 2. Filtrar Movimentações até a data final do mês com Lógica de Reset ---
        
        # Filtro padrão até o dia anterior
        df_neg_temp = df_neg[df_neg["Data do Negócio"] <= dia_anterior_str].copy()
        df_mov_temp = df_mov[df_mov['Data'] <= dia_anterior_str].copy()
        # Aplica o filtro de Reset PM
        if not data_reset_pm.empty:
            
            # 2a. Aplicar filtro para df_neg (Negócios)
            df_neg_filtrado = df_neg_temp.merge(data_reset_pm, on='Ticker', how='left')
            
            # Onde Data Reset PM é NaN, substitui por uma data antiga (para manter todas as transações)
            df_neg_filtrado['Data Reset PM'] = df_neg_filtrado['Data Reset PM'].fillna(pd.to_datetime('1900-01-01'))
            
            df_neg_filtrado['Data do Negócio DT'] = pd.to_datetime(df_neg_filtrado['Data do Negócio'])
            
            # Filtra: mantém a linha se a Data do Negócio for ESTRIAMENTE MAIOR que Data Reset PM
            # (Transações do dia da venda total (Data Reset PM) são removidas)
            df_neg_filtrado = df_neg_filtrado[
                df_neg_filtrado['Data do Negócio DT'] > df_neg_filtrado['Data Reset PM']
            ].drop(columns=['Data Reset PM', 'Data do Negócio DT'])
            
            # 2b. Aplicar filtro para df_mov (Outras Movimentações, como bônus ou subscrição)
            # O mesmo filtro deve ser aplicado se df_mov afeta a carteira e o custo de aquisição.
            df_mov_filtrado = df_mov_temp.merge(data_reset_pm, on='Ticker', how='left')
            df_mov_filtrado['Data Reset PM'] = df_mov_filtrado['Data Reset PM'].fillna(pd.to_datetime('1900-01-01'))

            df_mov_filtrado['Data DT'] = pd.to_datetime(df_mov_filtrado['Data'])
            
            df_mov_filtrado = df_mov_filtrado[
                df_mov_filtrado['Data DT'] > df_mov_filtrado['Data Reset PM']
            ].drop(columns=['Data Reset PM', 'Data DT'])
            
        else:
            # Se não houver histórico de vendas totais, usa o filtro original (sem merge, mais rápido)
            df_neg_filtrado = df_neg_temp
            df_mov_filtrado = df_mov_temp


        # --- 3. Consolidação e Cálculo de Lucro ---
        
        # Consolidar carteira até essa data com o histórico AJUSTADO
        df_carteira_atual = consolidar_carteira(
            df_mov_filtrado, df_neg_filtrado, desdobros, subscricoes, bonus, leilao, ajuste_grupamento
        )[['Ticker', 'Qtd Final', 'Total Investido', 'Qtd Vendida','Preço Médio Ajustado']]
        # Vendas naquele dia
        vendas_naquele_dia = vendas[vendas['Data do Negócio'] == data_final_mes]

        # Merge para calcular lucro
        df_lucro = df_carteira_atual.merge(vendas_naquele_dia, on='Ticker', how='inner')


        # Classificar tipo de venda
        df_lucro["tipo venda"] = np.where(
            # Qtd Final aqui é a posição antes da VENDA do dia. Se a quantidade vendida no dia 
            # é igual à Qtd Final que tínhamos, é uma venda total.
            df_lucro["Qtd Final"] == df_lucro["Quantidade"],
            "venda total",
            "venda parcial"
        )

        # ... (Seu código de cálculo de lucro permanece o mesmo e está correto)
        # Calcular lucro/prejuízo
        condicao_inf = np.isinf(df_lucro['Preço Médio Ajustado'])

        # 2. Use np.where para aplicar as duas lógicas de cálculo:
        df_lucro['lucro'] = np.where(
            # SE a condição for verdadeira (infinito):
            condicao_inf,
            # ENTÃO use esta fórmula (ajuste que você solicitou):
            df_lucro['Valor'] - df_lucro['Total Investido'],
            # SENÃO (se não for infinito):
            # Use a sua fórmula original
            df_lucro['Valor'] - (df_lucro['Quantidade'] * df_lucro['Preço Médio Ajustado'])
        )
        # Adicionar data
        df_lucro["Data do Negócio"] = data_final_mes.strftime('%Y-%m-%d')
        # Selecionar apenas as colunas que você quer consolidar
        df_lucro = df_lucro[["Data do Negócio", "Ticker", "lucro", "tipo venda",'Preço Médio Ajustado']]
        # Guardar no histórico
        historico_lucros.append(df_lucro)
        # Concatena tudo no final
        df_lucros_completo = pd.concat(historico_lucros, ignore_index=True)
        df_lucros_completo = df_lucros_completo.sort_values(by=["Data do Negócio", "Ticker"]).reset_index(drop=True)

    
    return df_lucros_completo

# ---------------- ROTAS ----------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Backend está rodando!"}

# Define o tempo máximo que o cache fica válido
CACHE_EXPIRATION = timedelta(hours=1)
_last_cache_time = datetime.min

# ===============================
# 🔹 Função cacheada
# ===============================
@lru_cache(maxsize=1)
def _gerar_carteira_cache():
    print("♻️  Recalculando carteira completa...")

    df_mov, df_neg, cnpj_b3 = carregar_dados()
    df_mov, df_neg = preparar_dados(df_mov, df_neg)
    desdobros, subscricoes, subscricoes_original, bonus, bonus_original, leilao, leilao_original = processar_eventos(df_mov)
    ajuste_grupamento = aplicar_grupamentos(df_neg)
    df_carteira = consolidar_carteira(df_mov, df_neg, desdobros, subscricoes, bonus, leilao, ajuste_grupamento)
    df_carteira_filtrada = df_carteira[df_carteira['Qtd Final'] > 0].copy()

    # ---- Cálculos resumidos ----
    df_lucros = calcular_lucros_vendas(df_neg, df_mov, desdobros, subscricoes, bonus, leilao, ajuste_grupamento)
    proventos_pivot_ir = calcular_proventos_ir(df_mov)
    df = df_carteira_filtrada.iloc[:-1].merge(proventos_pivot_ir, on="Ticker", how="left").fillna(0)

    # ---- Ajuste de nomes ----
    df = df.rename(columns={
        "Qtd Final": "quantidade",
        "Preço Médio Ajustado": "preco_medio",
        "Dividendo": "dividendos",
        "Juros Sobre Capital Próprio": "juros_sobre_capital_proprio",
        'Total Vendido': "total_investido"
    })
    cnpj_b3['Ticker'] = cnpj_b3['Ticker'].str.split(' ')

    # 2. "Explodir" a coluna Ticker para que cada item da lista ganhe uma linha própria
    # O Pandas irá repetir o CNPJ e a Razão Social para cada novo Ticker gerado
    cnpj_b3 = cnpj_b3.explode('Ticker')

    # 3. Limpeza adicional: remover espaços em branco que possam ter sobrado e linhas vazias
    cnpj_b3['Ticker'] = cnpj_b3['Ticker'].str.strip()
    cnpj_b3 = cnpj_b3[cnpj_b3['Ticker'] != '']
    
    json_final = gerar_json_ir(df, cnpj_b3, df_lucros)
    return json_final

    # ---- Retorno padronizado ----
    # return {
    #     "carteira": df_carteira_filtrada[[
    #         "Ticker", "preco_medio", "quantidade","dividendos",
    #         "juros_sobre_capital_proprio"
    #     ]].to_dict(orient="records"),
    #     "resumos": {
    #         "lucro_prejuizo_total": df_lucros['lucro'].sum()
    #     }
    # }

# ===============================
# 🔹 Rota principal
# ===============================
@app.get("/relatorio-ir")
def get_relatorio_ir():
    global _last_cache_time

    try:
        agora = datetime.now()
        if agora - _last_cache_time > CACHE_EXPIRATION:
            print(f"🕒 Cache expirado ({(agora - _last_cache_time).seconds//60} min). Recalculando...")
            _gerar_carteira_cache.cache_clear()
            _last_cache_time = agora

        return _gerar_carteira_cache()

    except Exception as e:
        print("🔥 ERRO NA ROTA /carteira 🔥")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

