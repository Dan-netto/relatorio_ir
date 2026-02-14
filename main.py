# ---------------- CONFIGURAÇÕES ----------------
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from fastapi.responses import JSONResponse
import os, traceback
from functools import lru_cache
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import exchange_calendars as ecals
import math
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

app = FastAPI(debug=True)

ano_fiscal='2025'
B3_CAL = ecals.get_calendar("BVMF")
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

def carregar_dados():
        df_mov = pd.read_sql("SELECT * FROM movimentacao_2026_02_12", con=engine)
        # df_neg = pd.read_sql("SELECT * FROM ldinvest_negociacao_21092025", con=engine)
        cnpj_b3 = pd.read_sql("SELECT * FROM cnpj_b3_total", con=engine)
        df_subscricao = pd.read_sql("SELECT * FROM base_precos_subscricao_total", con=engine)
        df_provisionados = pd.read_sql("SELECT * FROM df_provisionados", con=engine)
        incorporacoes_cvm = pd.read_sql("SELECT * FROM df_incorporacoes_cvm", con=engine)
        cnpj_complemento=pd.read_sql("SELECT * FROM cnpj_b3", con=engine)
        cnpj_complemento['Ticker'] = (
        cnpj_complemento['Ticker']
        .astype(str)
        .str.strip()
        .str.split(r'\s+')
        )

        # transforma cada item da lista em uma linha
        cnpj_complemento = cnpj_complemento.explode('Ticker', ignore_index=True)

        # limpa casos vazios (se existir)
        cnpj_complemento['Ticker'] = cnpj_complemento['Ticker'].str.strip()
        cnpj_complemento = cnpj_complemento[cnpj_complemento['Ticker'] != '']

        cnpj_b3=pd.concat([cnpj_b3, cnpj_complemento], ignore_index=True).drop_duplicates(subset=['CNPJ', 'Ticker'])

        return df_mov, cnpj_b3, df_subscricao,df_provisionados,incorporacoes_cvm

def preparar_dados(df_mov,df_subscricao,ano_fiscal):
    # for col in ['Preço', 'Valor', 'Quantidade']:
    #     df_neg[col] = limpar_valor(df_neg[col])
    for col in ['Preço unitário', 'Valor da Operação', 'Quantidade']:
        df_mov[col] = limpar_valor(df_mov[col])

    df_mov['Ticker'] = df_mov['Produto'].str.extract(r'^([A-Z0-9]+)')
    # df_neg['Ticker'] = df_neg['Código de Negociação'].apply(normalizar_ticker)

    # df_neg['Data do Negócio'] = pd.to_datetime(df_neg['Data do Negócio'], dayfirst=True, errors='coerce')
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], dayfirst=True, errors='coerce')

    df_mov_subscr = df_mov[df_mov['Movimentação'].isin(['Direitos de Subscrição - Exercido'])].copy()

    # Garantir que Data é datetime
    df_mov_subscr["Data"] = pd.to_datetime(df_mov_subscr["Data"])

    # Criar coluna Ano
    df_mov_subscr["Ano"] = df_mov_subscr["Data"].dt.year.astype(str)

    # Criar coluna semestre
    df_mov_subscr["semestre"] = np.where(
        df_mov_subscr["Data"].dt.month <= 6,
        "primeiro_semestre",
        "segundo_semestre"
    )
    # Converter preço para float (trocar vírgula por ponto)
    df_subscricao["Preco_Subscricao"] = (
    df_subscricao["Preco_Subscricao"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
    )

    df_mov_subscr['Ticker Raiz'] = df_mov_subscr['Ticker'].apply(raiz_ticker)
    df_subscricao['Ticker Raiz'] = df_subscricao['Ticker'].apply(raiz_ticker)
    df_subscricao = df_subscricao.drop(columns=["Ticker"])
    df_merge = df_mov_subscr.merge(
    df_subscricao,
    on=["Ticker Raiz", "Ano", "semestre"],
    how="left"
    )

    df_merge["Preço unitário"] = df_merge["Preço unitário"].fillna(
    df_merge["Preco_Subscricao"]
    )
    df_merge["Valor da Operação"] = df_merge["Valor da Operação"].fillna(
        df_merge["Preço unitário"] * df_merge["Quantidade"]
    )
    df_final = df_merge.drop(columns=["Preco_Subscricao"])

    df_final = df_final.drop(columns=["Ano", "semestre",'Ticker Raiz', 'Empresa'])

    df_mov_sem_subscr = df_mov[
    ~df_mov["Movimentação"].isin(["Direitos de Subscrição - Exercido"])
    ].copy()

    df_mov = pd.concat(
    [df_mov_sem_subscr, df_final],
    ignore_index=True,
    sort=False
    )

    data_fim = pd.to_datetime(f"{ano_fiscal}-12-31")
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], dayfirst=True)

    # Now the filter is much cleaner
    df_mov = df_mov[(df_mov['Data'] < data_fim)]

    return df_mov

def norm_str(s):
        s = s.astype(str).str.strip().str.lower()
        # remove acentos
        s = s.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        return s

def padroniza(df, origem,cols_alvo):
        # Garante existência das colunas alvo (cria se faltar)
        for c in cols_alvo:
            if c not in df.columns:
                df[c] = np.nan
        
        # Reordena e copia
        out = df[cols_alvo].copy()

        # Tipos
        out['Data do Negócio'] = pd.to_datetime(out['Data do Negócio'], errors='coerce', dayfirst=True)
        # Quantidade como inteiro se possível; senão, mantém numérico
        out['Quantidade'] = pd.to_numeric(out['Quantidade'], errors='coerce')
        # Se você tiver certeza que quantidade é sempre inteira: descomente a linha abaixo
        # out['Quantidade'] = out['Quantidade'].round().astype('Int64')

        out['Preço'] = pd.to_numeric(out['Preço'], errors='coerce')
        out['Valor'] = pd.to_numeric(out['Valor'], errors='coerce')

        # Normaliza Ticker
        out['Ticker'] = out['Ticker'].astype(str).str.strip().str.upper()

        # Normaliza Tipo de Movimentação (opcional, para consistência)
        out['Tipo de Movimentação'] = out['Tipo de Movimentação'].astype(str).str.strip()

        # Origem (para você saber de onde cada linha veio)
        out['Origem'] = origem
        return out

def clean_val(val):
        if pd.isna(val) or np.isinf(val):
            return 0.0
        return float(val)

def resolver_ticker_mae(ticker_sub, tickers_consolidado):
    """
    ticker_sub: str  -> ex: 'ITSA1', 'ITSA2'
    tickers_consolidado: iterable -> ex: consolidado['Ticker']
    """
    raiz = ticker_sub.rstrip('0123456789')
    sufixo = ticker_sub[len(raiz):]

    # Regra principal: subscrição
    if sufixo == '1' and f'{raiz}3' in tickers_consolidado:
        return f'{raiz}3'

    if sufixo == '2' and f'{raiz}4' in tickers_consolidado:
        return f'{raiz}4'


    # Último fallback: retorna o próprio
    return ticker_sub



######### CONVERSÃO MOVIMENTAÇÕES EM NEGOCIAÇÕES
def classificar_movimentacoes_v7(df_original):
    
    df = df_original.copy()
    
    # 1. Preparação de tipos e normalização
    df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
    df['Quantidade'] = pd.to_numeric(df['Quantidade'], errors='coerce').fillna(0).round(6)
    
    for col in ['Preço unitário', 'Valor da Operação']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).replace({'-': '0', 'nan': '0'}), errors='coerce').fillna(0)

    df['Mov_Norm'] = norm_str(df['Movimentação'])
    df['Ticker'] = df['Produto'].str.split(' ').str[0].str.strip().str.upper()

    # 2. Separar TLs e EMPs
    mask_tl = df['Mov_Norm'].eq('transferencia - liquidacao')
    mask_emp = df['Mov_Norm'].eq('emprestimo')
    
    # Criamos o banco de dados de empréstimos
    emp_df = df[mask_emp].copy().sort_values('Data')

    # 3. Agrupar TLs (caso a liquidação ocorra em várias linhas no mesmo dia)
    tliq = df[mask_tl].groupby(['Ticker', 'Data', 'Entrada/Saída'], as_index=False).agg({
        'Quantidade': 'sum', 
        'Preço unitário': 'mean',
        'Valor da Operação': 'sum'
    })


    # Aplicamos a verificação linha a linha nas TLs
    tliq['is_aluguel'] = tliq.apply(
    verificar_origem_aluguel, 
    axis=1, 
    args=(emp_df, 5)  # Passa o emp_df e a janela_dias como tupla
    )

    # 5. Filtragem Final
    df_reais = tliq[~tliq['is_aluguel']].copy()

    # 6. Data do Negócio e Tipo
    df_reais['Data do Negócio'] = subtrai_dois_dias_uteis_b3(df_reais['Data'])
    
    es_norm = norm_str(df_reais['Entrada/Saída'])
    df_reais['Tipo'] = np.where(es_norm.eq('credito'), 'Compra', 'Venda')

    df_reais.rename(columns={
        'Preço unitário': 'Preço',
        'Valor da Operação': 'Valor',
        'Tipo': 'Tipo de Movimentação'
    }, inplace=True)

    cols_finais = ['Ticker','Data do Negócio', 'Tipo de Movimentação', 'Quantidade', 'Preço', 'Valor']
    return df_reais[cols_finais].sort_values('Data do Negócio')

def subtrai_dois_dias_uteis_b3(dates_series):
    # 1. Normaliza para Series datetime
    if not isinstance(dates_series, pd.Series):
        dates_series = pd.Series(pd.to_datetime(dates_series))
    else:
        dates_series = pd.to_datetime(dates_series)

    def subtrair(data):
        if pd.isna(data):
            return pd.NaT

        # Se a data não for sessão, pega a sessão anterior
        if not B3_CAL.is_session(data):
            data = B3_CAL.previous_session(data)

        # Subtrai 2 sessões
        return B3_CAL.session_offset(data, -2)

    return dates_series.apply(subtrair)

def verificar_origem_aluguel(row,emp_df,janela_dias=5):
        # Filtra empréstimos do mesmo Ticker na janela de X dias anteriores à TL
        data_limite_inf = row['Data'] - pd.Timedelta(days=janela_dias)
        
        possiveis_emps = emp_df[
            (emp_df['Ticker'] == row['Ticker']) & 
            (emp_df['Data'] <= row['Data']) & 
            (emp_df['Data'] >= data_limite_inf)
        ]
        
        if possiveis_emps.empty:
            return False
            
        qtd_tl = row['Quantidade']
        qtds_individuais = possiveis_emps['Quantidade'].unique()
        soma_total_janela = possiveis_emps['Quantidade'].sum()
        
        # Filtro: É aluguel se...
        # A) A quantidade da TL bate com QUALQUER empréstimo individual da janela
        if qtd_tl in qtds_individuais:
            return True
        # B) A quantidade da TL bate com a SOMA de todos os empréstimos da janela
        if round(qtd_tl, 6) == round(soma_total_janela, 6):
            return True
        # C) Se o preço for zero, as chances de ser aluguel são altíssimas
        if row['Preço unitário'] == 0:
            return True
            
        return False


######## CONSOLIDAÇÃO DA CARTEIRA
def processar_fluxo_historico(df_neg, df_mov,cnpj_b3,ano_fiscal,data_eventos_provisionados,incorporacoes_cvm):
    tipos_eventos = ['Desdobro', 'Grupamento', 'Incorporação', 
                     'Direitos de Subscrição - Exercido', 'Bonificação em Ativos']

    tipos_eventos_neg=['Venda']
    
    fila_eventos = df_mov[df_mov['Movimentação'].isin(tipos_eventos)].copy()
    fila_eventos_neg = df_neg[df_neg['Tipo de Movimentação'].isin(tipos_eventos_neg)].copy()
    fila_eventos_provisionados=pd.to_datetime(data_eventos_provisionados, dayfirst=True)
    fila_eventos['Data_DT'] = pd.to_datetime(fila_eventos['Data'], dayfirst=True)
    fila_eventos_neg['Data_DT'] = pd.to_datetime(fila_eventos_neg['Data do Negócio'], dayfirst=True)
    datas_eventos = sorted(
    pd.concat([
        fila_eventos['Data_DT'],
        fila_eventos_neg['Data_DT'],
        pd.Series(fila_eventos_provisionados)
    ]).unique()
    )

    tickers_afetados_incorporacao = []
    historico_snapshots = []
    
    consolidado = pd.DataFrame(columns=[
        'Ticker', 'Qtd Compra', 'Total Compra', 'Qtd Vendida', 'Total Vendido', 
        'Ticker Raiz', 'Total Investido', 'Qtd Leiloada', 'Qtd_subscr', 'Quantidade_Desdobro','Qtd Bonus','Qtd incorporada'
    ])

    data_inicio = pd.to_datetime('1900-01-01')

    for data_evento in datas_eventos:
        mask_neg = (pd.to_datetime(df_neg['Data do Negócio'], dayfirst=True) >= data_inicio) & \
                   (pd.to_datetime(df_neg['Data do Negócio'], dayfirst=True) < data_evento)
        neg_periodo = df_neg[mask_neg]

        consolidado = novo_consolidar_carteira(neg_periodo, base_inicial=consolidado)

        # PASSO B: Processar Eventos do Dia (Fiel às suas fórmulas)
        eventos_dia = fila_eventos[fila_eventos['Data_DT'] == data_evento]
        
        if eventos_dia.empty:
            eventos_dia_neg = fila_eventos_neg[fila_eventos_neg['Data_DT'] == data_evento]
            if not eventos_dia_neg.empty:
                for _, ev in eventos_dia_neg.iterrows():
                    # break
                    ticker_original = ev['Ticker']
                    tickers_existentes = set(consolidado['Ticker'])
                    ticker_alvo = resolver_ticker_mae(ticker_original, tickers_existentes)
                    # Se o ticker_alvo (pai ou ele mesmo) não existe, criamos
                    if ticker_alvo not in consolidado['Ticker'].values:
                        nova_linha = pd.DataFrame([{c: 0 for c in consolidado.columns}])
                        nova_linha['Ticker'] = ticker_alvo
                        nova_linha['Ticker Raiz'] = raiz_ticker(ticker_alvo)
                        consolidado = pd.concat([consolidado, nova_linha], ignore_index=True)

                    idx = consolidado[consolidado['Ticker'] == ticker_alvo].index[0]
                    preco_medio_antes_da_venda=consolidacao_final(consolidado,tickers_afetados_incorporacao).query('Ticker == @ticker_alvo')['Preço Médio Ajustado'].values[0]
                    # print(f"Preço médio antes da venda: {preco_medio_antes_da_venda}")
                    if ev['Tipo de Movimentação'] == 'Venda':
                        investido = ev['Quantidade'] * preco_medio_antes_da_venda
                        consolidado.at[idx, 'Total Investido'] -= investido
            else:
                consolidado=consolidado.copy() # Apenas para manter a consistência do processo, mesmo que não haja eventos 
        else:
            for _, ev in eventos_dia.iterrows():
                
                ticker_original = ev['Ticker']
                tickers_existentes = set(consolidado['Ticker'])
                ticker_alvo = resolver_ticker_mae(ticker_original, tickers_existentes)
                
                # Se o ticker_alvo (pai ou ele mesmo) não existe, criamos
                if ticker_alvo not in consolidado['Ticker'].values:
                    nova_linha = pd.DataFrame([{c: 0 for c in consolidado.columns}])
                    nova_linha['Ticker'] = ticker_alvo
                    nova_linha['Ticker Raiz'] = raiz_ticker(ticker_alvo)
                    consolidado = pd.concat([consolidado, nova_linha], ignore_index=True)

                idx = consolidado[consolidado['Ticker'] == ticker_alvo].index[0]

                # --- Aplicação das Fórmulas ---
                if ev['Movimentação'] == 'Desdobro':
                    consolidado.at[idx, 'Quantidade_Desdobro'] += ev['Quantidade']
                    # consolidado.at[idx, 'Qtd Compra'] += ev['Quantidade']

                elif ev['Movimentação'] == 'Direitos de Subscrição - Exercido':
                    qty = ev['Quantidade']
                    qty = 0.0 if math.isnan(qty) else qty
                    pcy = ev['Preço unitário']
                    pcy = 0.0 if math.isnan(pcy) else pcy
                    investido = qty * pcy
                    consolidado.at[idx, 'Qtd_subscr'] += qty
                    consolidado.at[idx, 'Total Investido'] += investido # Soma ao custo histórico
                    # consolidado.at[idx, 'Qtd Compra'] += ev['Quantidade']

                elif ev['Movimentação'] == 'Bonificação em Ativos':
                    qtd_bonus = np.floor(ev['Quantidade'])
                    # investido_b = qtd_bonus * ev['Preço unitário'] # Custo atribuído pela empresa
                    consolidado.at[idx, 'Qtd Bonus'] += qtd_bonus
                    # consolidado.at[idx, 'Total Investido'] += investido_b # ISSO É IMPORTANTE
                    # consolidado.at[idx, 'Qtd Compra'] += qtd_bonus
                elif ev['Movimentação'] == 'Incorporação':
                    res_inc = incorporacao(consolidado, df_mov, cnpj_b3,incorporacoes_cvm)
                    tickers_afetados_incorporacao = res_inc[0] # Pega a lista de tickers "mortos"
                    ticker_sucessor = res_inc[1]
                    custo_total_acumulado = res_inc[2]
                    qtd_incorporada_b3 = res_inc[3]
                    # tickers_afetados_incorporacao,ticker_sucessor, custo_total_acumulado,qtd_incorporada_b3=incorporacao(consolidado, df_mov, cnpj_b3)
                    idx_sucessor = consolidado[consolidado['Ticker'] == ticker_sucessor].index[0]
                    consolidado.at[idx_sucessor, 'Qtd incorporada'] += qtd_incorporada_b3
                    consolidado.at[idx_sucessor, 'Total Investido'] += custo_total_acumulado


        snapshot = consolidado.copy()
        snapshot['Snapshot_Data'] = data_evento
        historico_snapshots.append(snapshot)

        data_inicio = data_evento
        
    # Convert the column to datetime first
    data_fim = pd.to_datetime(f"{ano_fiscal}-12-31")
    df_neg['Data do Negócio'] = pd.to_datetime(df_neg['Data do Negócio'], dayfirst=True)

    # Now the filter is much cleaner
    neg_finais = df_neg[(df_neg['Data do Negócio'] >= data_inicio) & 
                        (df_neg['Data do Negócio'] < data_fim)]
    
    
    if not neg_finais.empty:
        consolidado = novo_consolidar_carteira(neg_finais, base_inicial=consolidado)
        
    # PASSO EXTRA: Rendimentos (Fora do loop de custódia pois não alteram PM)
    consolidado = aplicar_rendimentos_finais(consolidado, df_mov)
    
    # Adicionamos o estado final (Hoje) ao histórico
    final_snap = consolidado.copy()
    final_snap['Snapshot_Data'] = data_fim
    historico_snapshots.append(final_snap)

    return tickers_afetados_incorporacao,historico_snapshots

def novo_consolidar_carteira(df_neg_periodo, base_inicial=None):

    colunas_finais = [
        'Ticker', 'Qtd Compra', 'Total Compra', 'Qtd Vendida', 'Total Vendido', 
        'Ticker Raiz', 'Total Investido', 'Qtd Leiloada', 'Qtd_subscr', 
        'Quantidade_Desdobro', 'Qtd Bonus','Qtd incorporada','Link PDF'
    ]
    
    # --- Processamento do Período Atual ---
    c = df_neg_periodo[df_neg_periodo['Tipo de Movimentação'] == 'Compra']
    compras = c.groupby('Ticker').agg({
        'Quantidade': 'sum',
        'Preço': lambda x: (x * c.loc[x.index, 'Quantidade']).sum()
    }).rename(columns={'Quantidade': 'Qtd Compra', 'Preço': 'Total Compra'})

    v = df_neg_periodo[df_neg_periodo['Tipo de Movimentação'] == 'Venda']
    vendas = v.groupby('Ticker').agg({
        'Quantidade': 'sum',
        'Preço': lambda x: (x * v.loc[x.index, 'Quantidade']).sum()
    }).rename(columns={'Quantidade': 'Qtd Vendida', 'Preço': 'Total Vendido'})

    df_periodo = compras.join(vendas, how='outer').fillna(0).reset_index()
    df_periodo['Ticker Raiz'] = df_periodo['Ticker'].apply(raiz_ticker)
    
    # O "Total Investido" do período é apenas o que foi comprado agora
    df_periodo['Total Investido'] = df_periodo['Total Compra']
    
    # Garantir que todas as colunas existam antes de somar
    for col in colunas_finais:
        if col not in df_periodo.columns:
            df_periodo[col] = 0.0

    # --- Soma com a Base Inicial (Acumulação) ---
    if base_inicial is not None and not base_inicial.empty:
        # Usamos apenas Ticker como índice para evitar conflitos de Ticker Raiz
        df_periodo = df_periodo.set_index('Ticker')
        base_inicial = base_inicial.set_index('Ticker')
        
        # Somamos os valores. O .add() manterá o que já existia na base (inclusive custos de eventos)
        # e somará com as novas compras/vendas.
        colunas_numericas = [c for c in colunas_finais if c not in ['Ticker', 'Ticker Raiz']]
        df_consolidado = base_inicial[colunas_numericas].add(df_periodo[colunas_numericas], fill_value=0.0)
        
        df = df_consolidado.reset_index()
        df['Ticker Raiz'] = df['Ticker'].apply(raiz_ticker)
    else:
        df = df_periodo

    df['Total Investido'] = np.where(
        df['Qtd Compra'] <= df['Qtd Vendida'], 
        0, 
        df['Total Investido'] # Aqui você deve garantir que esse valor seja o custo das compras
    )
    return df[colunas_finais]

def aplicar_rendimentos_finais(df_consolidado, df_mov):
    
    rendimentos = df_mov[df_mov['Movimentação'] == 'Rendimento'].copy()
    filtros_fii = ['FII', 'FDO', 'FUNDO DE INVESTIMENTO IMOBILIARIO']
    rendimentos['É FII'] = rendimentos['Produto'].str.upper().str.contains('|'.join(filtros_fii), na=False)
    rendimentos['Ticker'] = rendimentos['Produto'].str.extract(r'^([A-Z0-9]+)')
    rendimentos['Ticker Raiz'] = rendimentos['Ticker'].apply(raiz_ticker)

    # Rendimento FII
    rfii = rendimentos[rendimentos['É FII']].groupby('Ticker Raiz')['Valor da Operação'].sum().reset_index()
    rfii.columns = ['Ticker Raiz', 'Rendimento_fii']

    # Rendimento Ações
    raco = rendimentos[~rendimentos['É FII']].groupby('Ticker Raiz')['Valor da Operação'].sum().reset_index()
    raco.columns = ['Ticker Raiz', 'Rendimento_acoes']

    # Merge com o consolidado
    df_consolidado = df_consolidado.merge(rfii, on='Ticker Raiz', how='left').merge(raco, on='Ticker Raiz', how='left')

    fracao = df_mov[df_mov['Movimentação'] == 'Leilão de Fração'].copy()
    if not fracao.empty:
        leilao_map = fracao.groupby('Ticker')['Quantidade'].sum()
        # Soma na coluna existente
        df_consolidado['Qtd Leiloada'] = df_consolidado['Qtd Leiloada'].fillna(0) + \
                                        df_consolidado['Ticker'].map(leilao_map).fillna(0)
    

    return df_consolidado.fillna(0)

def incorporacao(df, df_mov, cnpj_b3,incorporacoes_cvm):

    cnpj_b3['Raiz'] = cnpj_b3['Ticker'].apply(raiz_ticker)
    cnpj_b3_unique = cnpj_b3.drop_duplicates(subset=['Raiz'])[['Raiz', 'CNPJ', 'Razão Social']]

    incorporacoes_na_b3 = df_mov[df_mov['Movimentação'] == 'Incorporação']
    incorporacoes_na_b3['Raiz'] = incorporacoes_na_b3['Ticker'].apply(raiz_ticker)
    incorporacoes_na_b3 = incorporacoes_na_b3.merge(cnpj_b3_unique, on='Raiz', how='left')
    df['Raiz'] = df['Ticker'].apply(raiz_ticker)
    df = df.merge(cnpj_b3_unique, on='Raiz', how='left')
    tickers_sucessores = incorporacoes_na_b3['Ticker'].unique().tolist()
    if not incorporacoes_na_b3.empty:
        for ticker_sucessor in tickers_sucessores:
            print(f"\n--- Analisando Incorporação para o Ticker: {ticker_sucessor} ---")

            # Busca o CNPJ do sucessor no seu mapeamento
            match_cnpj = cnpj_b3[cnpj_b3['Ticker'] == ticker_sucessor]
            if match_cnpj.empty:
                print(f"⚠️ Ticker {ticker_sucessor} não encontrado no mapeamento cnpj_b3. Pulando...")
                continue
        
            cnpj_sucessor = match_cnpj.iloc[0]['CNPJ']

            custo_total_acumulado = 0
    encontrou_e_processou = False # Flag para parar a busca em outros PDFs

    mask_sucessor = incorporacoes_cvm['CNPJ_Sucessor'] == cnpj_sucessor
    mask_incorporada = incorporacoes_cvm['CNPJ_Incorporada'] == cnpj_sucessor

    resultado = incorporacoes_cvm.loc[
            mask_sucessor | mask_incorporada
        ]

    cnpj_antecessor_encontrado = (
            resultado.apply(
                lambda row: (
                    row['CNPJ_Incorporada']
                    if row['CNPJ_Sucessor'] == cnpj_sucessor
                    else row['CNPJ_Sucessor']
                ),
                axis=1
            )
        ).values
    cnpj_antecessor_encontrado = df[df['CNPJ'].isin(cnpj_antecessor_encontrado)]['CNPJ'].unique().tolist()
    if cnpj_antecessor_encontrado:
            tickers_afetados = cnpj_b3[cnpj_b3['CNPJ'].isin(cnpj_antecessor_encontrado)]['Ticker'].unique().tolist()
            print(f"🎯 Match encontrado! O PDF cita {tickers_afetados}.")

            # Processar cada ticker da empresa que está "morrendo"
            for ticker_ante in tickers_afetados:
                dados_ante = df[df['Ticker'] == ticker_ante]
                
                if not dados_ante.empty:
                    valor_investido = dados_ante['Total Investido'].sum()
                    custo_total_acumulado += valor_investido
                    
                    # Log de confirmação
                    print(f"💰 Migrando R$ {valor_investido:.2f} de {ticker_ante} para {ticker_sucessor}")
                    
                    # Ativamos a flag para não processar outros PDFs deste ticker sucessor
                    encontrou_e_processou = True

    # Se após rodar todos os PDFs (ou dar break) tivermos custo, finalizamos
    if encontrou_e_processou:
        print(f"✅ Sucesso: Total consolidado para {ticker_sucessor}: R$ {custo_total_acumulado:.2f}")
        qtd_inc_b3 = int(incorporacoes_na_b3[incorporacoes_na_b3['Ticker'] == ticker_sucessor]['Quantidade'].sum())
        return (tickers_afetados,ticker_sucessor, custo_total_acumulado, qtd_inc_b3)
    
    return None # Caso nenhum PDF tenha dado match

def consolidacao_final(df,tickers_afetados_incorporacao):
    # Migra para o sucessor
    # df.loc[df['Ticker'] == ticker_sucessor, 'Total Investido'] += custo_total_acumulado
    
    # df.loc[df['Ticker'] == ticker_sucessor, 'qtd_incorporada'] = qtd_incorporada_b3

    # Remove o antecessor da custódia atual
    # df = df[df['Ticker'] != ticker_antecessor]
        
    df['Qtd Final'] = df['Qtd Compra'] + df['Qtd Bonus'] + df['Qtd_subscr'] + df['Quantidade_Desdobro'] + df['Qtd incorporada'].fillna(0) - df['Qtd Vendida']

    qtd_total_historico = (
    df['Qtd Compra'].fillna(0) +
    df['Qtd Bonus'].fillna(0) +
    df['Qtd_subscr'].fillna(0) +
    df['Quantidade_Desdobro'].fillna(0)+
    df['Qtd incorporada'].fillna(0)
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
    
    tickers_set = {str(t).strip().upper() for t in tickers_afetados_incorporacao}
    mask = df['Ticker'].astype(str).str.strip().str.upper().isin(tickers_set)
    
    # Converte a coluna 'Qtd Final' para numérico se não estiver
    df['Qtd Final'] = pd.to_numeric(df['Qtd Final'], errors='coerce')

    # Aplica a regra: se ticker afetado → Qtd Final = 0.0
    df.loc[mask, 'Qtd Final'] = 0.0

    df_carteira = df[['Ticker', 'Qtd Final', 'Total Investido', 'Preço Médio Ajustado','Qtd Vendida', 'Total Vendido']].sort_values('Ticker')

    return(df_carteira)

def cisao(df_carteira,df_mov):
    # aplicar cisão AZEV4 → AZEV4 + AZTE3
    cisao = df_mov[df_mov['Movimentação'] == 'Cisão'].copy()

    cisao_agrupada = cisao.groupby('Ticker')['Quantidade'].sum().reset_index()

    # Filtramos apenas os que não existem no df_carteira
    faltantes = cisao_agrupada[~cisao_agrupada['Ticker'].isin(df_carteira['Ticker'])]

    # 2. Se houver tickers faltantes, cria o DF novo
    if not faltantes.empty:
        df_novos = pd.DataFrame({
            'Ticker': faltantes['Ticker'],
            'Qtd Final': faltantes['Quantidade'],
            'Total Investido': 0.0,
            'Preço Médio Ajustado': 0.0,
            'Qtd Vendida': 0,
            'Total Vendido': 0.0
        })
         # 3. Juntar ao dataframe original
        df_carteira = pd.concat([df_carteira, df_novos], ignore_index=True).sort_values('Ticker')

   

    return(df_carteira)

######## Provento recebido no ano fiscal (para IR)
def calcular_proventos_ir(df_mov,ano_fiscal):
    # 1. Preparação das datas
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], dayfirst=True)
    
    inicio_ir = pd.Timestamp(year=int(ano_fiscal), month=1, day=1)
    fim_ir = pd.Timestamp(year=int(ano_fiscal), month=12, day=31)

    # 2. Filtro de Período (Ano-Calendário) e Tipos de Proventos
    tipos_proventos = ['Juros Sobre Capital Próprio', 'Dividendo','Reembolso','Rendimento']
    
    mask = (
        (df_mov['Data'] >= inicio_ir) & 
        (df_mov['Data'] <= fim_ir) & 
        (df_mov['Movimentação'].isin(tipos_proventos))
    )
    
    df_filtrado = df_mov[mask].copy()

    if df_filtrado.empty:
        return pd.DataFrame(columns=['Ticker', 'Dividendo', 'Juros Sobre Capital Próprio'])

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

    df_pivot['Ticker Raiz'] = df_pivot['Ticker'].apply(raiz_ticker)

    df_pivot_rend = aplicar_rendimentos_finais(df_pivot, df_filtrado)

    df_pivot_rend.drop(columns=['Ticker Raiz'], inplace=True)

    # 5. Garantir que todas as colunas existam (mesmo que o usuário não tenha recebido um dos tipos)
    for col in tipos_proventos:
        if col not in df_pivot_rend.columns:
            df_pivot_rend[col] = 0.0

    return df_pivot_rend


####### Vendas e Lucros para IR (com lógica de reset de PM a cada venda total)
def calcular_lucros_vendas_novo(df_neg, df_mov,df_carteira_final_historico,tickers_afetados_incorporacao):


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

    historico_lucros = []
    historico_preco_medio_venda= []
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

        data_final_mes_dt = pd.to_datetime(data_final_mes)

        # --- O PULO DO GATO: BUSCAR SNAPSHOT ANTERIOR ---
        # Filtramos os snapshots que ocorreram ANTES da data desta venda
        snapshots_anteriores = [s for s in df_carteira_final_historico if s['Snapshot_Data'].max() < data_final_mes_dt]

        if snapshots_anteriores:
            # Pegamos o snapshot mais próximo (o último da lista filtrada)
            base_snapshot = snapshots_anteriores[-1].copy()
            data_base = base_snapshot['Snapshot_Data'].max()
        else:
            # Se não houver snapshot (venda antes de qualquer evento societário), base vazia
            base_snapshot = pd.DataFrame(columns=['Ticker', 'Qtd Compra', 'Total Investido', 'Ticker Raiz'])
            data_base = pd.to_datetime('1900-01-01')

        # --- CONSOLIDAR APENAS O INTERVALO (Snapshot até a Venda) ---
        # Filtramos negociações que ocorreram APÓS o snapshot e ANTES da venda atual
        mask_intervalo = (pd.to_datetime(df_neg['Data do Negócio'], dayfirst=True) >= data_base) & \
                         (pd.to_datetime(df_neg['Data do Negócio'], dayfirst=True) < data_final_mes_dt)
        neg_intervalo = df_neg[mask_intervalo]
        
        # Consolidar carteira até essa data com o histórico AJUSTADO
        df_carteira_atual = novo_consolidar_carteira(neg_intervalo, base_inicial=base_snapshot)

        df_carteira_atual = consolidacao_final(df_carteira_atual,tickers_afetados_incorporacao)
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
        historico_preco_medio_venda.append(df_carteira_atual)
        historico_lucros.append(df_lucro)
        # Concatena tudo no final
        df_lucros_completo = pd.concat(historico_lucros, ignore_index=True)
        df_lucros_completo = df_lucros_completo.sort_values(by=["Data do Negócio", "Ticker"]).reset_index(drop=True)

    
    return df_lucros_completo

####### Drill down
def filtra_pos_venda_total(df, data_col,cutoff_por_ticker):
    out = df.merge(cutoff_por_ticker.rename('cutoff'),
                left_on='Ticker', right_index=True, how='left')
    # Mantém linhas sem cutoff (NaT) OU com data > cutoff
    mask = out['cutoff'].isna() | (out[data_col] > out['cutoff'])
    out = out.loc[mask].drop(columns=['cutoff'])
    return out

def filtra_pre_venda_total(df, data_col,cutoff_por_ticker):
    out = df.merge(cutoff_por_ticker.rename('cutoff'),
                left_on='Ticker', right_index=True, how='left')
    # Mantém linhas sem cutoff (NaT) OU com data < cutoff
    mask = out['cutoff'].isna() | (out[data_col] <= out['cutoff'])
    out = out.loc[mask].drop(columns=['cutoff'])
    return out

def historico_vendas(df_vendas,df_neg,df_mov,df_lucros_novo):
    tickers_atuais=df_vendas['Ticker']
    tickers_set = pd.Index([str(t).strip().upper() for t in tickers_atuais])
    # Converte datas (usando dayfirst=True por padrão BR)
    df_neg['Data do Negócio'] = pd.to_datetime(df_neg['Data do Negócio'], errors='coerce', dayfirst=True)
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], errors='coerce', dayfirst=True)

    df_mov['Ticker'] = df_mov['Ticker'].apply(
        lambda x: resolver_ticker_mae(x, tickers_set)
    )

    # ---------- 2) Filtro inicial por tickers ----------
    df_neg_f = df_neg[df_neg['Ticker'].isin(tickers_set)].copy()
    df_mov_f = df_mov[df_mov['Ticker'].isin(tickers_set)].copy()

    # ---------- 3) Remover movimentações específicas em df_mov ----------
    # Remove "Empréstimo" e "Transferência - Liquidação"
    df_mov_f['Movimentação'] = df_mov_f['Movimentação'].astype(str).str.strip()
    tipos_incluir = {'Bonificação em Ativos','Cisão','Desdobro','Direitos de Subscrição - Exercido'}
    # tipos_excluir_neg = {'Venda'}
    df_mov_f = df_mov_f[df_mov_f['Movimentação'].isin(tipos_incluir)].copy()
    # df_neg_f = df_neg_f[~df_neg_f['Tipo de Movimentação'].isin(tipos_excluir_neg)].copy()

    # ---------- 4) Preparar cutoffs com base em 'venda total' ----------
    # Mantém apenas vendas totais por ticker presentes em tickers_atual
    mask_venda_total = df_lucros_novo['tipo venda'].astype(str).str.strip().str.lower().isin(['venda total', 'venda parcial'])
    df_lucros_vt = df_lucros_novo[mask_venda_total & df_lucros_novo['Ticker'].isin(tickers_set)].copy()

    # Para cada Ticker, pega a data MAIS RECENTE de 'venda total'
    cutoff_por_ticker = df_lucros_vt.groupby('Ticker', as_index=True)['Data do Negócio'].max()

    df_neg_f = filtra_pre_venda_total(df_neg_f, 'Data do Negócio',cutoff_por_ticker)
    df_mov_f = filtra_pre_venda_total(df_mov_f, 'Data',cutoff_por_ticker)

    # ---------- 5) Renomear colunas no df_mov e selecionar colunas ----------
    df_mov_f = df_mov_f.rename(columns={'Preço unitário': 'Preço',
                                        'Valor da Operação':'Valor',
                                          'Movimentação':'Tipo de Movimentação',
                                          'Data':'Data do Negócio'})

    # Seleção final de colunas:
    # Observação: mantenho "Ticker" para você identificar a qual ativo cada linha pertence.
    # Se quiser EXCLUSIVAMENTE as colunas solicitadas, basta remover 'Ticker' das listas abaixo.
    cols_neg_final = [c for c in ['Data do Negócio','Ticker','Tipo de Movimentação','Quantidade', 'Preço', 'Valor'] if c in df_neg_f.columns]
    cols_mov_final = [c for c in ['Data do Negócio','Ticker','Tipo de Movimentação','Quantidade', 'Preço', 'Valor','Link_PDF'] if c in df_mov_f.columns]
   

    df_neg_final = df_neg_f.loc[:, cols_neg_final].fillna(0.0).copy()
    df_mov_final = df_mov_f.loc[:, cols_mov_final].copy()

    # 2. Identificamos apenas as colunas numéricas (int e float)
    cols_numericas = df_mov_final.select_dtypes(include=['number']).columns

    # 3. Aplicamos o fillna apenas nessas colunas
    df_mov_final[cols_numericas] = df_mov_final[cols_numericas].fillna(0.0)


    # 1) Colunas alvo e garantia de existência
    cols_alvo_neg = ['Data do Negócio', 'Ticker', 'Tipo de Movimentação', 'Quantidade', 'Preço', 'Valor']
    cols_alvo_mov= ['Data do Negócio', 'Ticker', 'Tipo de Movimentação', 'Quantidade', 'Preço', 'Valor','Link_PDF']

    
    neg_hist = padroniza(df_neg_final, origem='Negociação', cols_alvo=cols_alvo_neg)
    mov_hist = padroniza(df_mov_final, origem='Movimentação', cols_alvo=cols_alvo_mov)

    # 2) Concatena
    df_hist = pd.concat([neg_hist, mov_hist], ignore_index=True)

    # 3) Remove linhas sem data (se houver)
    df_hist = df_hist[~df_hist['Data do Negócio'].isna()].copy()

    # 4) Ordenação: por Ticker, Data, e desempate por Origem (opcional)
    #    Se preferir priorizar Negociação antes de Movimentação no mesmo dia:
    categoria_origem = pd.CategoricalDtype(categories=['Negociação', 'Movimentação'], ordered=True)
    df_hist['Origem'] = df_hist['Origem'].astype(categoria_origem)

    df_hist = df_hist.sort_values(by=['Ticker', 'Data do Negócio', 'Origem'], ascending=[True, True, True]).reset_index(drop=True)

    # 5) (Opcional) Se quiser uma coluna "Sequência" por Ticker
    df_hist['Sequência'] = df_hist.groupby('Ticker').cumcount() + 1
    
    return(df_hist)

def historico_negociacoes(df_ir,df_neg,df_mov,df_lucros_novo):
    tickers_atuais=df_ir['Ticker']
    tickers_set = pd.Index([str(t).strip().upper() for t in tickers_atuais])
    # Converte datas (usando dayfirst=True por padrão BR)
    df_neg['Data do Negócio'] = pd.to_datetime(df_neg['Data do Negócio'], errors='coerce', dayfirst=True)
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], errors='coerce', dayfirst=True)

    df_mov['Ticker'] = df_mov['Ticker'].apply(
        lambda x: resolver_ticker_mae(x, tickers_set)
    )

    # ---------- 2) Filtro inicial por tickers ----------
    df_neg_f = df_neg[df_neg['Ticker'].isin(tickers_set)].copy()
    df_mov_f = df_mov[df_mov['Ticker'].isin(tickers_set)].copy()

    # ---------- 3) Remover movimentações específicas em df_mov ----------
    # Remove "Empréstimo" e "Transferência - Liquidação"
    df_mov_f['Movimentação'] = df_mov_f['Movimentação'].astype(str).str.strip()
    tipos_incluir = {'Bonificação em Ativos','Cisão','Desdobro','Direitos de Subscrição - Exercido'}
    df_mov_f = df_mov_f[df_mov_f['Movimentação'].isin(tipos_incluir)].copy()

    # ---------- 4) Preparar cutoffs com base em 'venda total' ----------
    # Mantém apenas vendas totais por ticker presentes em tickers_atual
    mask_venda_total = df_lucros_novo['tipo venda'].astype(str).str.strip().str.lower().eq('venda total')
    df_lucros_vt = df_lucros_novo[mask_venda_total & df_lucros_novo['Ticker'].isin(tickers_set)].copy()

    # Para cada Ticker, pega a data MAIS RECENTE de 'venda total'
    cutoff_por_ticker = df_lucros_vt.groupby('Ticker', as_index=True)['Data do Negócio'].max()

    df_neg_f = filtra_pos_venda_total(df_neg_f, 'Data do Negócio',cutoff_por_ticker)
    df_mov_f = filtra_pos_venda_total(df_mov_f, 'Data',cutoff_por_ticker)

    # ---------- 5) Renomear colunas no df_mov e selecionar colunas ----------
    df_mov_f = df_mov_f.rename(columns={'Preço unitário': 'Preço',
                                        'Valor da Operação':'Valor',
                                          'Movimentação':'Tipo de Movimentação',
                                          'Data':'Data do Negócio'})

    # Seleção final de colunas:
    # Observação: mantenho "Ticker" para você identificar a qual ativo cada linha pertence.
    # Se quiser EXCLUSIVAMENTE as colunas solicitadas, basta remover 'Ticker' das listas abaixo.
    cols_neg_final = [c for c in ['Data do Negócio','Ticker','Tipo de Movimentação','Quantidade', 'Preço', 'Valor'] if c in df_neg_f.columns]
    cols_mov_final = [c for c in ['Data do Negócio','Ticker','Tipo de Movimentação','Quantidade', 'Preço', 'Valor','Link_PDF'] if c in df_mov_f.columns]
   

    df_neg_final = df_neg_f.loc[:, cols_neg_final].fillna(0.0).copy()
    df_mov_final = df_mov_f.loc[:, cols_mov_final].copy()

    # 2. Identificamos apenas as colunas numéricas (int e float)
    cols_numericas = df_mov_final.select_dtypes(include=['number']).columns

    # 3. Aplicamos o fillna apenas nessas colunas
    df_mov_final[cols_numericas] = df_mov_final[cols_numericas].fillna(0.0)


    # 1) Colunas alvo e garantia de existência
    cols_alvo_neg = ['Data do Negócio', 'Ticker', 'Tipo de Movimentação', 'Quantidade', 'Preço', 'Valor']
    cols_alvo_mov= ['Data do Negócio', 'Ticker', 'Tipo de Movimentação', 'Quantidade', 'Preço', 'Valor','Link_PDF']

    
    neg_hist = padroniza(df_neg_final, origem='Negociação', cols_alvo=cols_alvo_neg)
    mov_hist = padroniza(df_mov_final, origem='Movimentação', cols_alvo=cols_alvo_mov)

    # 2) Concatena
    df_hist = pd.concat([neg_hist, mov_hist], ignore_index=True)

    # 3) Remove linhas sem data (se houver)
    df_hist = df_hist[~df_hist['Data do Negócio'].isna()].copy()

    # 4) Ordenação: por Ticker, Data, e desempate por Origem (opcional)
    #    Se preferir priorizar Negociação antes de Movimentação no mesmo dia:
    categoria_origem = pd.CategoricalDtype(categories=['Negociação', 'Movimentação'], ordered=True)
    df_hist['Origem'] = df_hist['Origem'].astype(categoria_origem)

    df_hist = df_hist.sort_values(by=['Ticker', 'Data do Negócio', 'Origem'], ascending=[True, True, True]).reset_index(drop=True)

    # 5) (Opcional) Se quiser uma coluna "Sequência" por Ticker
    df_hist['Sequência'] = df_hist.groupby('Ticker').cumcount() + 1
    
    return(df_hist)

def historico_proventos(df_proventos, df_mov, ano_fiscal):
    # 1) Preparação dos Tickers
    tickers_atuais = df_proventos['Ticker']
    tickers_set = set([str(t).strip().upper() for t in tickers_atuais])
    ano = int(ano_fiscal)

    # 2) Limpeza e Filtro de Datas
    df_mov = df_mov.copy()
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], errors='coerce', dayfirst=True)
    
    inicio = pd.Timestamp(year=ano, month=1, day=1)
    fim = pd.Timestamp(year=ano, month=12, day=31, hour=23, minute=59, second=59)
    
    # Filtra tickers e período
    df_mov_f = df_mov[
        (df_mov['Ticker'].isin(tickers_set)) & 
        (df_mov['Data'].between(inicio, fim))
    ].copy()

    # 3) Categorização dos Proventos
    # Mapeamento para separar o que é cada coisa
    def categorizar_provento(mov):
        mov = str(mov).strip()
        if 'Dividendo' in mov:
            return 'Dividendo'
        elif 'Juros Sobre Capital Próprio' in mov or 'JCP' in mov:
            return 'JCP'
        elif 'Rendimento' in mov:
            return 'Rendimento'
        elif 'Reembolso' in mov:
            return 'Reembolso'
        else:
            return 'Outros'

    df_mov_f['Categoria_IR'] = df_mov_f['Movimentação'].apply(categorizar_provento)

    # 4) Filtrar apenas o que interessa (removemos compras, vendas e eventos de custódia)
    tipos_validos = ['Dividendo', 'JCP', 'Rendimento','Reembolso']
    df_mov_f = df_mov_f[df_mov_f['Categoria_IR'].isin(tipos_validos)].copy()

    # 5) Padronização de Colunas
    df_mov_f = df_mov_f.rename(columns={
        'Preço unitário': 'Preço',
        'Valor da Operação': 'Valor',
        'Movimentação': 'Tipo de Movimentação',
        'Data': 'Data do Negócio'
    })

    # Seleção de colunas finais para o drill-down
    cols_finais = ['Data do Negócio', 'Ticker', 'Tipo de Movimentação', 'Quantidade', 'Preço', 'Valor', 'Categoria_IR']
    df_hist_prov = df_mov_f[cols_finais].fillna(0.0).copy()

    # 6) Ordenação e Sequência
    df_hist_prov = df_hist_prov.sort_values(by=['Ticker', 'Data do Negócio']).reset_index(drop=True)
    df_hist_prov['Sequência'] = df_hist_prov.groupby('Ticker').cumcount() + 1
    
    return df_hist_prov

######### proventos provisionados
def extrair_data_provisionado(tickers_provisionados,df_provisionados):
    tickers_set = pd.Index([str(t).strip().upper() for t in tickers_provisionados])
    df_provisionado_ajustado = df_provisionados[df_provisionados['Ticker'].isin(tickers_set)].copy()
    data_eventos_provisionados2 = df_provisionado_ajustado['data_com'].dt.strftime('%Y-%d-%m').unique().tolist()
    return data_eventos_provisionados2, df_provisionado_ajustado

def proventos_provisionados(df_carteira_final_historico,df_provisionado_ajustado):
    # juntar todos os dfs da lista
    colunas_necessarias = ['Snapshot_Data','Ticker', 'Qtd Compra', 'Qtd Bonus', 'Qtd_subscr', 'Quantidade_Desdobro', 'Qtd incorporada', 'Qtd Vendida']

    df_historico = pd.concat(
        [df[colunas_necessarias] for df in df_carteira_final_historico],
        ignore_index=True
    )

    df_historico['Qtd Final'] = df_historico['Qtd Compra'] + df_historico['Qtd Bonus'] + df_historico['Qtd_subscr'] + df_historico['Quantidade_Desdobro'] + df_historico['Qtd incorporada'].fillna(0) - df_historico['Qtd Vendida']

    # garantir datetime

    df_merge = df_provisionado_ajustado.merge(
        df_historico,
        left_on=['Ticker', 'data_com'],
        right_on=['Ticker', 'Snapshot_Data'],
        how='left'
    )
    df_merge['valor'] = pd.to_numeric(df_merge['valor'], errors='coerce')
    df_merge['valor_provisionado'] = df_merge['Qtd Final'] * df_merge['valor']

    df_final_provisionado = df_merge.groupby(['Ticker', 'data_com','tipo'], as_index=False)['valor_provisionado'].sum()
    df_final_provisionado=df_final_provisionado[df_final_provisionado['valor_provisionado'] > 0].copy()
    return df_final_provisionado

######## JSON para FRONT

def gerar_json_ir(df_ir, df_proventos, cnpj_b3, df_lucros, df_historico_negociacoes, df_historico_proventos,df_historico_vendas, ano_fiscal):
    """
    df_ir: Contém ['Ticker', 'Qtd Final', 'Total Investido', 'Preço Médio Ajustado']
    df: Contém os totais agregados ['Ticker', 'dividendos', 'juros_sobre_capital_proprio', 'Reembolso', 'Rendimento_fii', 'Rendimento_acoes']
    """
    tickers_no_ir = df_ir['Ticker'].unique()
    tickers_com_proventos = df_historico_proventos['Ticker'].unique()
    todos_tickers = pd.Series(np.union1d(tickers_no_ir, tickers_com_proventos), name='Ticker')
    df_base = pd.DataFrame(todos_tickers)
    # --- 1. PREPARAÇÃO DA CARTEIRA PRINCIPAL ---
    # Unificamos dados de custódia, totais de proventos e informações de CNPJ
    df_consolidado = pd.merge(df_base, df_ir, on='Ticker', how='left')
    df_consolidado = pd.merge(df_consolidado, df_proventos, on='Ticker', how='left', suffixes=('', '_drop'))
    df_consolidado = pd.merge(df_consolidado, cnpj_b3, on='Ticker', how='left')
    
    df_consolidado['CNPJ'] = df_consolidado['CNPJ'].fillna("00.000.000/0000-00")
    df_consolidado['Razão Social'] = df_consolidado['Razão Social'].fillna("Razão Social não encontrada")

    lista_carteira_detalhada = []
    # --- 2. LOOP PARA CRIAR O DRILL-DOWN POR TICKER ---
    for _, row in df_consolidado.iterrows():
        ticker = row['Ticker']
        # Detalhes de Negociações (Explicação do Preço Médio)
        # Filtramos o histórico e ordenamos pela sequência
        neg_ticker = df_historico_negociacoes[df_historico_negociacoes['Ticker'] == ticker].sort_values('Sequência')
        detalhes_neg = neg_ticker.copy()
        detalhes_neg['Data do Negócio'] = detalhes_neg['Data do Negócio'].dt.strftime('%d/%m/%Y')
        
        # Dentro do loop de tickers na função gerar_json_ir:

        prov_ticker = df_historico_proventos[df_historico_proventos['Ticker'] == ticker].copy()
        prov_ticker['Data do Negócio'] = pd.to_datetime(prov_ticker['Data do Negócio']).dt.strftime('%d/%m/%Y')
        
        # Filtramos as listas de detalhes para o front-end
        detalhes_dividendos = prov_ticker[prov_ticker['Categoria_IR'] == 'Dividendo'].to_dict(orient='records')
        detalhes_jcp = prov_ticker[prov_ticker['Categoria_IR'] == 'JCP'].to_dict(orient='records')
        detalhes_rendimentos = prov_ticker[prov_ticker['Categoria_IR'] == 'Rendimento'].to_dict(orient='records')
        detalhes_reembolsos = prov_ticker[prov_ticker['Categoria_IR'] == 'Reembolso'].to_dict(orient='records')
        
        quantidade_final = clean_val(row.get('Qtd Final'))
        # Montagem do objeto do Ticker
        item_carteira = {
            "ticker": ticker,
            "cnpj": row['CNPJ'],
            "razao_social": row['Razão Social'],
            "possui_custodia_final": quantidade_final > 0,
            "custodia": {
                "quantidade_final": quantidade_final,
                "total_investido": clean_val(row.get('Total Investido')),
                "preco_medio_ajustado": clean_val(row.get('Preço Médio Ajustado'))
            },
            "totais_proventos": {
                "dividendos": clean_val(row.get('Dividendo')),
                "jcp": clean_val(row.get('Juros Sobre Capital Próprio')),
                "reembolso": clean_val(row.get('Reembolso')),
                "rendimento_fii": clean_val(row.get('Rendimento_fii')),
                "rendimento_acoes": clean_val(row.get('Rendimento_acoes'))
            },
            "drill_down_negociacoes": detalhes_neg.to_dict(orient='records'),
            "drill_down_dividendos": detalhes_dividendos,
            "drill_down_jcp": detalhes_jcp,
            "drill_down_rendimentos": detalhes_rendimentos,
            "drill_down_reembolsos": detalhes_reembolsos
        }
        
        lista_carteira_detalhada.append(item_carteira)
        
    # --- 3. PREPARAÇÃO DOS LUCROS MENSAIS (Renda Variável) ---
    df_lucros['Data do Negócio'] = pd.to_datetime(df_lucros['Data do Negócio'])
    df_lucros_ir = df_lucros[df_lucros['Data do Negócio'].dt.year == int(ano_fiscal)].copy()
    
    meses_nomes = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun", 
                   7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
    
    lista_lucros_mensais = []

    for m in range(1, 13):
        # Filtra lucros do mês
        lucros_mes = df_lucros_ir[df_lucros_ir['Data do Negócio'].dt.month == m]
        valor_total_lucro = clean_val(lucros_mes['lucro'].sum())
        
        detalhes_vendas_mes = []
        
        # Para cada ticker vendido no mês, buscamos o histórico que justifica o PM
        for _, venda in lucros_mes.iterrows():
            ticker_vendido = venda['Ticker']
            data_venda = venda['Data do Negócio']
            
            # Filtra o histórico de eventos RELEVANTES (Compras/Eventos) até a data desta venda
            # Usamos o df_historico_vendas que você mencionou
            historico_venda = df_historico_vendas[
                (df_historico_vendas['Ticker'] == ticker_vendido) & 
                (pd.to_datetime(df_historico_vendas['Data do Negócio']) <= data_venda)
            ].sort_values('Sequência').copy()
            
            # Seleciona apenas as colunas solicitadas
            colunas_view = ['Data do Negócio', 'Ticker', 'Tipo de Movimentação', 'Quantidade', 'Preço', 'Valor']
            historico_view = historico_venda[colunas_view].fillna(0)
            historico_view['Data do Negócio'] = pd.to_datetime(historico_view['Data do Negócio']).dt.strftime('%d/%m/%Y')

            detalhes_vendas_mes.append({
                "ticker": ticker_vendido,
                "data_venda": data_venda.strftime('%d/%m/%Y'),
                "lucro_apurado": clean_val(venda['lucro']),
                "tipo_venda": venda.get('tipo venda', 'venda'),
                "drill_down_historico": historico_view.to_dict(orient='records')
            })

        lista_lucros_mensais.append({
            "mes": m,
            "mes_nome": meses_nomes[m],
            "lucro_total": valor_total_lucro,
            "vendas": detalhes_vendas_mes # Aqui está o novo drill down das vendas
        })

    # --- 4. JSON FINAL ---
    return {
        "ano_referencia": ano_fiscal,
        "resumo_anual_lucros": lista_lucros_mensais,
        "carteira": lista_carteira_detalhada
    }

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

    df_mov, cnpj_b3,df_subscricao,df_provisionados,incorporacoes_cvm= carregar_dados()
    df_mov= preparar_dados(df_mov,df_subscricao,ano_fiscal)
    df_neg = classificar_movimentacoes_v7(df_mov)
    tickers_provisionados = df_neg['Ticker'].unique()
    datas_provisionadas,df_provisionado_ajustado=extrair_data_provisionado(tickers_provisionados,df_provisionados)
    tickers_afetados_incorporacao,df_carteira_final_historico = processar_fluxo_historico(df_neg,df_mov,cnpj_b3,ano_fiscal,datas_provisionadas,incorporacoes_cvm)
    df_carteira_final=df_carteira_final_historico[-1]
    df_final_provisionado = proventos_provisionados(df_carteira_final_historico,df_provisionado_ajustado)
    # ---- Cálculos resumidos ----
    df_lucros = calcular_lucros_vendas_novo(df_neg, df_mov,df_carteira_final_historico,tickers_afetados_incorporacao)
    
    df_carteira = consolidacao_final(df_carteira_final,tickers_afetados_incorporacao)
    
    df_carteira = cisao(df_carteira,df_mov)
    

    proventos_pivot_ir = calcular_proventos_ir(df_mov,ano_fiscal)
    df = df_carteira.iloc[:-1].merge(proventos_pivot_ir, on="Ticker", how="left")
    df_ir=df[df['Qtd Final'] > 0][['Ticker','Qtd Final','Total Investido','Preço Médio Ajustado']].copy()

    df_proventos = df[['Ticker','Dividendo','Juros Sobre Capital Próprio','Reembolso','Rendimento_fii','Rendimento_acoes']].fillna(0)

    df_historico_negociacoes=historico_negociacoes(df_ir,df_neg,df_mov,df_lucros)
    df_historico_negociacoes['Link_PDF'] = df_historico_negociacoes['Link_PDF'].fillna('-')
    df_historico_proventos=historico_proventos(df_proventos,df_mov,ano_fiscal)
    df_vendas=df_carteira[df_carteira['Qtd Final'] == 0]
    df_historico_vendas=historico_vendas(df_vendas,df_neg,df_mov,df_lucros)
        

    # ---- Ajuste de nomes ----
    # df = df.rename(columns={
    #     "Qtd Final": "quantidade",
    #     "Preço Médio Ajustado": "preco_medio",
    #     "Dividendo": "dividendos",
    #     "Juros Sobre Capital Próprio": "juros_sobre_capital_proprio",
    #     'Total Vendido': "total_investido",
    #     'Qtd Vendida': "quantidade_vendida",
    # })
    
    
    json_final = gerar_json_ir(df_ir, df_proventos, cnpj_b3, df_lucros, df_historico_negociacoes, df_historico_proventos,df_historico_vendas, ano_fiscal)
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

