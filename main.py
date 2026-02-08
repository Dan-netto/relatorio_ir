# ---------------- CONFIGURAÇÕES ----------------
import pandas as pd
from pandas import (DataFrame,Series,concat,date_range,read_sql,to_datetime,to_numeric,Timestamp,DateOffset,NA,MultiIndex)
import numpy as np
from numpy import (floor as np_floor,inf as np_inf,isinf as np_isinf,where as np_where)
from scipy.optimize import newton, brentq
from datetime import datetime, timedelta
import logging
from fastapi.responses import JSONResponse
import os, re, traceback
from functools import lru_cache
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import zipfile
import io
import pdfplumber
from collections import Counter
import unicodedata
import exchange_calendars as ecals
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
        df_mov = pd.read_sql("SELECT * FROM ldinvest_movimentacao_21092025", con=engine)
        # df_neg = pd.read_sql("SELECT * FROM ldinvest_negociacao_21092025", con=engine)
        cnpj_b3 = pd.read_sql("SELECT * FROM cnpj_b3_total", con=engine)
        df_subscricao = pd.read_sql("SELECT * FROM base_precos_subscricao_total", con=engine)
        return df_mov, cnpj_b3, df_subscricao

def preparar_dados(df_mov,df_subscricao):
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

    return df_mov

def norm_str(s):
        s = s.astype(str).str.strip().str.lower()
        # remove acentos
        s = s.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        return s

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

def processar_fluxo_historico(df_neg, df_mov,cnpj_b3,ano_fiscal):
    tipos_eventos = ['Desdobro', 'Grupamento', 'Incorporação', 
                     'Direitos de Subscrição - Exercido', 'Bonificação em Ativos']
    
    fila_eventos = df_mov[df_mov['Movimentação'].isin(tipos_eventos)].copy()
    fila_eventos['Data_DT'] = pd.to_datetime(fila_eventos['Data'], dayfirst=True)
    datas_eventos = sorted(fila_eventos['Data_DT'].unique())
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

        if not neg_periodo.empty:
            consolidado = novo_consolidar_carteira(neg_periodo, base_inicial=consolidado)

        # PASSO B: Processar Eventos do Dia (Fiel às suas fórmulas)
        eventos_dia = fila_eventos[fila_eventos['Data_DT'] == data_evento]
        
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
                investido = ev['Quantidade'] * ev['Preço unitário']
                consolidado.at[idx, 'Qtd_subscr'] += ev['Quantidade']
                consolidado.at[idx, 'Total Investido'] += investido # Soma ao custo histórico
                # consolidado.at[idx, 'Qtd Compra'] += ev['Quantidade']

            elif ev['Movimentação'] == 'Bonificação em Ativos':
                qtd_bonus = np.floor(ev['Quantidade'])
                # investido_b = qtd_bonus * ev['Preço unitário'] # Custo atribuído pela empresa
                consolidado.at[idx, 'Qtd Bonus'] += qtd_bonus
                # consolidado.at[idx, 'Total Investido'] += investido_b # ISSO É IMPORTANTE
                # consolidado.at[idx, 'Qtd Compra'] += qtd_bonus
            elif ev['Movimentação'] == 'Incorporação':
                res_inc = incorporacao(consolidado, df_mov, cnpj_b3)
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
    final_snap['Snapshot_Data'] = pd.Timestamp.now()
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

def normalizar_texto(texto):
    """Remove acentos e deixa tudo em minúsculo para comparação robusta."""
    if not texto: return ""
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    return texto.lower().strip()

def limpar_razao_social(nome):
    """Remove sufixos comuns que podem variar no PDF (S.A., S/A, Ltda)."""
    nome = normalizar_texto(nome)
    # Remove S.A, S/A, SA, LTDA, S.A., etc.
    nome = re.sub(r'\b(s/?a|ltda|participacoes|energia|brasil)\b', '', nome)
    # Pega apenas as duas primeiras palavras (geralmente o nome 'core' da empresa)
    palavras = nome.split()
    return " ".join(palavras[:2]) if len(palavras) >= 2 else nome

def mapear_incorporada_pela_carteira_v2(texto_pdf, df_minha_carteira, cnpj_sucessor):
    """
    1. Tenta por CNPJ (Precisão Máxima).
    2. Se falhar, tenta por Palavras-Chave da Razão Social (Segurança).
    """
    texto_norm = normalizar_texto(texto_pdf)
    
    # --- PASSO 1: BUSCA POR CNPJ ---
    regex_cnpj = r"(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})"
    cnpjs_no_texto = set(re.findall(regex_cnpj, texto_pdf))
    cnpjs_no_texto.discard(cnpj_sucessor)
    
    cnpjs_carteira = set(df_minha_carteira['CNPJ'].unique())
    encontrados_cnpj = cnpjs_no_texto.intersection(cnpjs_carteira)
    
    if encontrados_cnpj:
        return list(encontrados_cnpj)[0]

    # --- PASSO 2: BUSCA POR RAZÃO SOCIAL (Se o CNPJ falhar) ---
    print("CNPJ não encontrado. Iniciando busca por Razão Social...")
    
    # Filtra a carteira para não buscar a própria empresa sucessora
    df_busca = df_minha_carteira[df_minha_carteira['CNPJ'] != cnpj_sucessor]
    for _, row in df_busca.iterrows():
        
        nome_original = row['Razão Social']
        termo_busca = limpar_razao_social(nome_original)
        
        # Usamos regex com \b para garantir que estamos achando a palavra inteira
        # Ex: evitar que 'AES' dê match em 'AESTHETIC'
        if re.search(rf"\b{re.escape(termo_busca)}\b", texto_norm):
            print(f"🎯 Match por Nome: '{termo_busca}' encontrado (Ref: {nome_original})")
            return row['CNPJ']
            
    return None

def incorporacao(df, df_mov, cnpj_b3):

    cnpj_b3['Raiz'] = cnpj_b3['Ticker'].apply(raiz_ticker)
    cnpj_b3_unique = cnpj_b3.drop_duplicates(subset=['Raiz'])[['Raiz', 'CNPJ', 'Razão Social']]

    incorporacoes_na_b3 = df_mov[df_mov['Movimentação'] == 'Incorporação']
    incorporacoes_na_b3['Raiz'] = incorporacoes_na_b3['Ticker'].apply(raiz_ticker)
    incorporacoes_na_b3 = incorporacoes_na_b3.merge(cnpj_b3_unique, on='Raiz', how='left')
    df['Raiz'] = df['Ticker'].apply(raiz_ticker)
    df = df.merge(cnpj_b3_unique, on='Raiz', how='left')
    tickers_sucessores = incorporacoes_na_b3['Ticker'].unique().tolist()
    if not incorporacoes_na_b3.empty:
        ANO_ALVO = incorporacoes_na_b3['Data'].dt.year.min()

        URL_CVM = f"http://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_{ANO_ALVO}.zip"
        print(f"Lendo base da CVM para {ANO_ALVO}...")
        r = requests.get(URL_CVM)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_nome = [f for f in z.namelist() if f.endswith('.csv')][0]
        df_cvm = pd.read_csv(z.open(csv_nome), sep=';', encoding='ISO-8859-1')

        # Filtros base da CVM para documentos de incorporação
        cond_docs = (df_cvm['Assunto'].str.contains("Incorporação", case=False, na=False)) & \
                    (df_cvm['Categoria'].isin(['Aviso aos Acionistas', 'Fato Relevante']))
        df_cvm_filtrado = df_cvm[cond_docs].copy()

        for ticker_sucessor in tickers_sucessores:
            print(f"\n--- Analisando Incorporação para o Ticker: {ticker_sucessor} ---")

            # Busca o CNPJ do sucessor no seu mapeamento
            match_cnpj = cnpj_b3[cnpj_b3['Ticker'] == ticker_sucessor]
            if match_cnpj.empty:
                print(f"⚠️ Ticker {ticker_sucessor} não encontrado no mapeamento cnpj_b3. Pulando...")
                continue
        
            cnpj_sucessor = match_cnpj.iloc[0]['CNPJ']
        
            # Filtra na base da CVM apenas documentos deste Ticker (Sucessor)
            docs_do_ticker = df_cvm_filtrado[df_cvm_filtrado['CNPJ_Companhia'] == cnpj_sucessor]
        
            if docs_do_ticker.empty:
                print(f"❌ Nenhum documento de incorporação encontrado na CVM para o CNPJ {cnpj_sucessor}")
                continue

            custo_total_acumulado = 0
    encontrou_e_processou = False # Flag para parar a busca em outros PDFs

    for _, row_doc in docs_do_ticker.iterrows():
        if encontrou_e_processou:
            break # Interrompe o loop de PDFs se já resolvemos a migração

        print(f"📄 Analisando documento: {row_doc['Assunto']}...")
        texto_pdf = ler_pdf_da_url(row_doc['Link_Download'])
        
        cnpj_antecessor_encontrado = mapear_incorporada_pela_carteira_v2(
            texto_pdf, 
            df_minha_carteira=df[df['CNPJ'].notna()][['CNPJ', 'Razão Social']],
            cnpj_sucessor=cnpj_sucessor
        )
            
        if cnpj_antecessor_encontrado:
            tickers_afetados = cnpj_b3[cnpj_b3['CNPJ'] == cnpj_antecessor_encontrado]['Ticker'].unique().tolist()
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

def ler_pdf_da_url(url):
    """Baixa o PDF e lê o conteúdo sem salvar em disco."""
    try:
        response = requests.get(url, timeout=15)
        # Transforma os bytes do download em um arquivo 'virtual' na memória
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            texto = ""
            for pagina in pdf.pages:
                texto += pagina.extract_text() + "\n"
        return texto
    except Exception as e:
        print(f"Erro ao ler PDF da URL: {e}")
        return ""

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
    
def gerar_json_ir(df, cnpj_b3, df_lucros,ano_fiscal):
    """
    Consolida os dados para o Relatório de IR.
    df_carteira_filtrada: Contém quantidade, preco_medio, total_investido, Dividendo, Juros Sobre Capital Próprio
    cnpj_b3: Tabela de de-para Ticker -> CNPJ e Razão Social
    df_lucros: Histórico de vendas com coluna 'lucro' e 'Data do Negócio'
    """

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
    df_lucros_ir = df_lucros[df_lucros['Data do Negócio'].dt.year == int(ano_fiscal)].copy()
    
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
        "ano_referencia": ano_fiscal,
        "carteira_ir": lista_carteira,
        "lucros_mensais": lista_lucros_mensais
    }

def calcular_proventos_ir(df_mov,ano_fiscal):
    # 1. Preparação das datas
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], dayfirst=True)
    
    inicio_ir = pd.Timestamp(year=int(ano_fiscal), month=1, day=1)
    fim_ir = pd.Timestamp(year=int(ano_fiscal), month=12, day=31)

    # 2. Filtro de Período (Ano-Calendário) e Tipos de Proventos
    tipos_proventos = ['Juros Sobre Capital Próprio', 'Dividendo','Reembolso']
    
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
        historico_lucros.append(df_lucro)
        # Concatena tudo no final
        df_lucros_completo = pd.concat(historico_lucros, ignore_index=True)
        df_lucros_completo = df_lucros_completo.sort_values(by=["Data do Negócio", "Ticker"]).reset_index(drop=True)

    
    return df_lucros_completo

def buscar_documentos_custo_cisao(ticker_mae, cnpj_mae, data_evento, df_cvm_anual):
    """
    Retorna apenas os documentos com altíssima probabilidade de conter 
    o fator de custo de uma cisão específica.
    """
    # Converter data para datetime para criar a janela
    data_dt = pd.to_datetime(data_evento)
    inicio_janela = data_dt - pd.Timedelta(days=15) # 15 dias antes
    fim_janela = data_dt + pd.Timedelta(days=60)    # Até 60 dias depois (comum demorar)
    
    df_busca = df_cvm_anual.copy()
    df_busca['Data_Referencia'] = df_busca['Data_Referencia'].astype(str).str.strip()

    # 2. Conversão especificando o formato exato
    df_busca['Data_Referencia_DT'] = pd.to_datetime(
        df_busca['Data_Referencia'], 
        format='%Y-%m-%d', 
        errors='coerce' # Se houver algo bizarro, vira NaT em vez de dar erro
    )

    # 1. Filtro Inicial: Empresa correta e Janela de tempo
    mask = (df_busca['CNPJ_Companhia'] == cnpj_mae) & \
            (pd.to_datetime(df_busca['Data_Referencia_DT']) >= inicio_janela) & \
            (pd.to_datetime(df_busca['Data_Referencia_DT']) <= fim_janela)

    docs_empresa = df_busca[mask].copy()
    # 2. Filtro de Categorias Relevantes (Fato Relevante e Comunicado ao Mercado)
    # Na CVM: 'Fato Relevante' e 'Comunicado ao Mercado'
    categorias_alvo = ['Fato Relevante', 'Comunicado ao Mercado', 'Aviso aos Acionistas']
    docs_empresa = docs_empresa[docs_empresa['Categoria'].isin(categorias_alvo)]

    # 3. Busca por Palavras-Chave de "Custo" (Ouro do documento)
    # Termos que indicam que o documento fala de IR/Custo
    termos_ouro = r'(custo|aquisicao|imposto|renda|alocacao|rateio|proporcao)'
    
    # Criamos uma coluna de relevância
    docs_empresa['Score'] = docs_empresa['Assunto'].str.contains(termos_ouro, case=False, na=False).astype(int)
    
    # Adicionamos bônus se o assunto for especificamente "Custo de Aquisição"
    docs_empresa.loc[docs_empresa['Assunto'].str.contains('custo', case=False, na=False), 'Score'] += 2

    # Retorna apenas os que têm score > 0, ordenados pelos mais prováveis
    return docs_empresa[docs_empresa['Score'] > 0].sort_values(by='Score', ascending=False)

def extrair_pontuacao_cnpjs(texto_pdf, cnpj_incorporadora):
    """
    Analisa o texto e retorna um dicionário com CNPJ: Pontuação.
    """
    # Regex para CNPJ (flexível para espaços e prefixos)
    regex_cnpj = r"(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})"
    cnpjs_encontrados = re.findall(regex_cnpj, texto_pdf)
    
    # Se não achar nada, retorna vazio
    if not cnpjs_encontrados:
        return {}

    # 1. Limpeza: Remover o CNPJ da própria incorporadora
    # (Usamos strip e padronização para evitar erros de comparação)
    candidatos_brutos = [c.strip() for c in cnpjs_encontrados if c.strip() != cnpj_incorporadora]
    
    # 2. Contagem de frequência
    contagem = Counter(candidatos_brutos)
    
    scores = {}
    for i, cnpj in enumerate(candidatos_brutos):
        if cnpj not in scores:
            # Pontuação baseada na frequência
            scores[cnpj] = contagem[cnpj] * 2
            
            # Bônus de posição (os primeiros citados costumam ser a incorporada)
            if i == 0: scores[cnpj] += 10
            elif i == 1: scores[cnpj] += 5
            
    return scores

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
    
def filtra_pos_venda_total(df, data_col,cutoff_por_ticker):
    out = df.merge(cutoff_por_ticker.rename('cutoff'),
                left_on='Ticker', right_index=True, how='left')
    # Mantém linhas sem cutoff (NaT) OU com data > cutoff
    mask = out['cutoff'].isna() | (out[data_col] > out['cutoff'])
    out = out.loc[mask].drop(columns=['cutoff'])
    return out

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

    # Fallback para UNIT
    if f'{raiz}11' in tickers_consolidado:
        return f'{raiz}11'

    # Último fallback: retorna o próprio
    return ticker_sub

def historico_negociacoes(df_ir,df_neg,df_mov,df_lucros_novo):
    tickers_atuais=df_ir['Ticker']
    tickers_set = pd.Index([str(t).strip().upper() for t in tickers_atuais])
    # Converte datas (usando dayfirst=True por padrão BR)
    df_neg['Data do Negócio'] = pd.to_datetime(df_neg['Data do Negócio'], errors='coerce', dayfirst=True)
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], errors='coerce', dayfirst=True)

    df_mov['Ticker_Ajustado'] = df_mov['Ticker'].apply(
        lambda x: resolver_ticker_mae(x, tickers_set)
    )

    # ---------- 2) Filtro inicial por tickers ----------
    df_neg_f = df_neg[df_neg['Ticker'].isin(tickers_set)].copy()
    df_mov_f = df_mov[df_mov['Ticker_Ajustado'].isin(tickers_set)].copy()

    # ---------- 3) Remover movimentações específicas em df_mov ----------
    # Remove "Empréstimo" e "Transferência - Liquidação"
    df_mov_f['Movimentação'] = df_mov_f['Movimentação'].astype(str).str.strip()
    tipos_incluir = {'Compra','Bonificação em Ativos','Cisão0','Desdobro','Direitos de Subscrição - Exercido'}
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

def historico_proventos(df_ir,df_mov,df_lucros_novo,ano_fiscal):

    tickers_atuais=df_ir['Ticker']
    tickers_set = pd.Index([str(t).strip().upper() for t in tickers_atuais])

    ano = int(ano_fiscal)

    # 2) Converte a coluna 'Data' para datetime (robusto a formatos) e com dayfirst=True
    df_mov = df_mov.copy()
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], errors='coerce', dayfirst=True)

    # 3) Define limites do ano fiscal (início e fim INCLUSIVOS)
    inicio = pd.Timestamp(year=ano, month=1, day=1)
    fim = pd.Timestamp(year=ano, month=12, day=31, hour=23, minute=59, second=59)

    # 4) Aplica o filtro
    mask = df_mov['Data'].between(inicio, fim, inclusive='both')
    # Converte datas (usando dayfirst=True por padrão BR)
    df_mov['Data'] = pd.to_datetime(df_mov['Data'], errors='coerce', dayfirst=True)
    
    


    # ---------- 2) Filtro inicial por tickers ----------
    df_mov_f = df_mov[df_mov['Ticker'].isin(tickers_set)].copy()

    # ---------- 3) Remover movimentações específicas em df_mov ----------
    # Remove "Empréstimo" e "Transferência - Liquidação"
    df_mov_f['Movimentação'] = df_mov_f['Movimentação'].astype(str).str.strip()
    tipos_excluir = {'Empréstimo', 'Transferência - Liquidação','Cisão','Grupamento','Transferência','Bonificação em Ativos','Atualização',
                     'Fração em Ativos','Desdobro','Reembolso'}
    df_mov_f = df_mov_f[~df_mov_f['Movimentação'].isin(tipos_excluir)].copy()

    # ---------- 4) Preparar cutoffs com base em 'venda total' ----------
    # Mantém apenas vendas totais por ticker presentes em tickers_atual
    mask_venda_total = df_lucros_novo['tipo venda'].astype(str).str.strip().str.lower().eq('venda total')
    df_lucros_vt = df_lucros_novo[mask_venda_total & df_lucros_novo['Ticker'].isin(tickers_set)].copy()

    # Para cada Ticker, pega a data MAIS RECENTE de 'venda total'
    cutoff_por_ticker = df_lucros_vt.groupby('Ticker', as_index=True)['Data do Negócio'].max()

    df_mov_f = filtra_pos_venda_total(df_mov_f, 'Data',cutoff_por_ticker)

    # ---------- 5) Renomear colunas no df_mov e selecionar colunas ----------
    df_mov_f = df_mov_f.rename(columns={'Preço unitário': 'Preço',
                                        'Valor da Operação':'Valor',
                                          'Movimentação':'Tipo de Movimentação',
                                          'Data':'Data do Negócio'})

    # Seleção final de colunas:
    # Observação: mantenho "Ticker" para você identificar a qual ativo cada linha pertence.
    # Se quiser EXCLUSIVAMENTE as colunas solicitadas, basta remover 'Ticker' das listas abaixo.
    cols_mov_final = [c for c in ['Data do Negócio','Ticker','Tipo de Movimentação','Quantidade', 'Preço', 'Valor'] if c in df_mov_f.columns]
   
    df_mov_final = df_mov_f.loc[:, cols_mov_final].fillna(0.0).copy()

    # 1) Colunas alvo e garantia de existência
    cols_alvo = ['Data do Negócio', 'Ticker', 'Tipo de Movimentação', 'Quantidade', 'Preço', 'Valor']

    df_hist_prov = padroniza(df_mov_final, origem='Movimentação', cols_alvo=cols_alvo) 

    # 3) Remove linhas sem data (se houver)
    df_hist_prov = df_hist_prov[~df_hist_prov['Data do Negócio'].isna()].copy()

    # 4) Ordenação: por Ticker, Data, e desempate por Origem (opcional)
    #    Se preferir priorizar Negociação antes de Movimentação no mesmo dia:
    categoria_origem = pd.CategoricalDtype(categories=['Negociação', 'Movimentação'], ordered=True)
    df_hist_prov['Origem'] = df_hist_prov['Origem'].astype(categoria_origem)

    df_hist_prov = df_hist_prov.sort_values(by=['Ticker', 'Data do Negócio', 'Origem'], ascending=[True, True, True]).reset_index(drop=True)

    # 5) (Opcional) Se quiser uma coluna "Sequência" por Ticker
    df_hist_prov['Sequência'] = df_hist_prov.groupby('Ticker').cumcount() + 1
    return(df_hist_prov)

def agrupar_emprestimos_proximos(df_emp):
            df = df_emp.copy()
            
            # 1. Garantir ordenação para comparar linhas subsequentes
            df = df.sort_values(['Ticker', 'Data'])
            
            # 2. Identificar quebras: Se mudou o Ticker OU se a diferença de dias > 1
            # diff() calcula a diferença para a linha de cima
            diff_dias = df.groupby('Ticker')['Data'].diff().dt.days
            
            # Se diff_dias > 1, marca como True (nova sequência)
            nova_sequencia = (diff_dias > 1) | (diff_dias.isna())
            
            # 3. Criar o ID do grupo usando a soma acumulada (cumsum)
            # Cada vez que 'nova_sequencia' é True, o número do grupo aumenta
            df['grupo_id'] = nova_sequencia.cumsum()
            
            # 4. Agrupar de fato
            df_agrupado = df.groupby(['grupo_id', 'Ticker'], as_index=False).agg({
                'Data': 'max',             # Mantém a maior data
                'Quantidade': 'sum',       # Soma as quantidades
                '_Qty_Match': 'sum'        # Soma o campo de match também
            })
            
            return df_agrupado.drop(columns=['grupo_id'])

def gerar_json_ir(df_ir, df, cnpj_b3, df_lucros, df_historico_negociacoes, df_historico_proventos, ano_fiscal):
    """
    df_ir: Contém ['Ticker', 'Qtd Final', 'Total Investido', 'Preço Médio Ajustado']
    df: Contém os totais agregados ['Ticker', 'dividendos', 'juros_sobre_capital_proprio', 'Reembolso', 'Rendimento_fii', 'Rendimento_acoes']
    """
    
    # --- 1. PREPARAÇÃO DA CARTEIRA PRINCIPAL ---
    # Unificamos dados de custódia, totais de proventos e informações de CNPJ
    df_consolidado = pd.merge(df_ir, df, on='Ticker', how='left')
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
        
        # Detalhes de Proventos (Explicação dos Dividendos Totais)
        # Aqui filtramos apenas o que caiu no ano fiscal selecionado
        prov_ticker = df_historico_proventos[
            (df_historico_proventos['Ticker'] == ticker) & 
            (pd.to_datetime(df_historico_proventos['Data do Negócio']).dt.year == int(ano_fiscal))
        ].sort_values('Sequência')
        
        detalhes_prov = prov_ticker.copy()
        detalhes_prov['Data do Negócio'] = pd.to_datetime(detalhes_prov['Data do Negócio']).dt.strftime('%d/%m/%Y')

        # Montagem do objeto do Ticker
        item_carteira = {
            "ticker": ticker,
            "cnpj": row['CNPJ'],
            "razao_social": row['Razão Social'],
            "custodia": {
                "quantidade_final": float(row['Qtd Final']),
                "total_investido": float(row['Total Investido']),
                "preco_medio_ajustado": float(row['Preço Médio Ajustado'])
            },
            "totais_proventos": {
                "dividendos": float(row.get('dividendos', 0)),
                "jcp": float(row.get('juros_sobre_capital_proprio', 0)),
                "reembolso": float(row.get('Reembolso', 0)),
                "rendimento_fii": float(row.get('Rendimento_fii', 0)),
                "rendimento_acoes": float(row.get('Rendimento_acoes', 0))
            },
            "drill_down_negociacoes": detalhes_neg.to_dict(orient='records'),
            "drill_down_proventos": detalhes_prov.to_dict(orient='records')
        }
        
        lista_carteira_detalhada.append(item_carteira)

    # --- 3. PREPARAÇÃO DOS LUCROS MENSAIS (Renda Variável) ---
    df_lucros['Data do Negócio'] = pd.to_datetime(df_lucros['Data do Negócio'])
    df_lucros_ir = df_lucros[df_lucros['Data do Negócio'].dt.year == int(ano_fiscal)].copy()
    
    meses_nomes = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun", 
                   7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
    
    lucros_por_mes = df_lucros_ir.groupby(df_lucros_ir['Data do Negócio'].dt.month)['lucro'].sum().to_dict()
    
    lista_lucros_mensais = [
        {"mes": m, "mes_nome": meses_nomes[m], "lucro": float(lucros_por_mes.get(m, 0.0))}
        for m in range(1, 13)
    ]

    # --- 4. JSON FINAL ---
    return {
        "ano_referencia": ano_fiscal,
        "resumo_anual_lucros": lista_lucros_mensais,
        "carteira": lista_carteira_detalhada
    }

# 4. Nova Lógica de Verificação (Individual e Soma)

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

    df_mov, cnpj_b3,df_subscricao= carregar_dados()
    df_mov= preparar_dados(df_mov,df_subscricao)

    df_neg = classificar_movimentacoes_v7(df_mov)

    tickers_afetados_incorporacao,df_carteira_final_historico = processar_fluxo_historico(df_neg,df_mov,cnpj_b3,ano_fiscal)
    df_carteira_final=df_carteira_final_historico[-1]
    # desdobros, subscricoes, subscricoes_original, bonus, bonus_original, leilao, leilao_original = processar_eventos(df_mov)
    # ajuste_grupamento = aplicar_grupamentos(df_mov, df_neg, desdobros, subscricoes, bonus, leilao, cnpj_b3)
    # df = consolidar_carteira(df_neg, desdobros, subscricoes, bonus, leilao)
    # ticker_sucessor, custo_total_acumulado,incorporacoes_na_b3=incorporacao(df, df_mov, cnpj_b3)
    df_carteira = consolidacao_final(df_carteira_final,tickers_afetados_incorporacao)
    # df_carteira = df_carteira[~df_carteira['Ticker'].isin([a['Ticker'] for a in ajuste_grupamento])]
        
    # df_carteira = pd.concat([df_carteira, pd.DataFrame(ajuste_grupamento)], ignore_index=True)
    df_carteira = cisao(df_carteira,df_mov)
    # df_carteira_filtrada = df_carteira[df_carteira['Qtd Final'] > 0].copy()

    # ---- Cálculos resumidos ----
    df_lucros_novo = calcular_lucros_vendas_novo(df_neg, df_mov,df_carteira_final_historico,tickers_afetados_incorporacao)
    # df_lucros = calcular_lucros_vendas(df_neg, df_mov, desdobros, subscricoes, bonus, leilao, ajuste_grupamento,cnpj_b3)

    proventos_pivot_ir = calcular_proventos_ir(df_mov,ano_fiscal)
    df = df_carteira.iloc[:-1].merge(proventos_pivot_ir, on="Ticker", how="left")
    df_ir=df[df['Qtd Final'] > 0][['Ticker','Qtd Final','Total Investido','Preço Médio Ajustado']].copy()

    df = df[['Ticker','Dividendo','Juros Sobre Capital Próprio','Reembolso','Rendimento_fii','Rendimento_acoes']].fillna(0)

    df_historico_negociacoes=historico_negociacoes(df_ir,df_neg,df_mov,df_lucros_novo)
    df_historico_negociacoes['Link_PDF'] = df_historico_negociacoes['Link_PDF'].fillna('-')
    df_historico_proventos=historico_proventos(df_ir,df_mov,df_lucros_novo,ano_fiscal)

        

    # ---- Ajuste de nomes ----
    df = df.rename(columns={
        "Qtd Final": "quantidade",
        "Preço Médio Ajustado": "preco_medio",
        "Dividendo": "dividendos",
        "Juros Sobre Capital Próprio": "juros_sobre_capital_proprio",
        'Total Vendido': "total_investido",
        'Qtd Vendida': "quantidade_vendida",
    })
    
    
    json_final = gerar_json_ir(df_ir, df, cnpj_b3, df_lucros_novo, df_historico_negociacoes, df_historico_proventos, ano_fiscal)
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

