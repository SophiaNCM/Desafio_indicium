# =========================================
# 1. IMPORTS
# =========================================
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# =========================================
# 2. CARREGAMENTO DOS DADOS
# =========================================
clientes = pd.read_csv('./csv/dim_clientes.csv')
produtos = pd.read_csv('./csv/dim_produtos.csv')
vendas = pd.read_csv('./csv/fatos_vendas.csv')



# =========================================
# 3. MATRIZ USUÁRIO × PRODUTO (BINÁRIA)
# =========================================

# Criar coluna de interação binária
vendas['interacao'] = 1

# Pivot para matriz usuário-produto
matriz_usuario_produto = vendas.pivot_table(
    index='id_client',
    columns='id_product',
    values='interacao',
    aggfunc='max',   # garante presença (1)
    fill_value=0     # ausência (0)
)
df_matriz_usuario_produto = pd.DataFrame(matriz_usuario_produto)
print("\nMatriz Usuário x Produto:")
print(df_matriz_usuario_produto )


# =========================================
# 4. SIMILARIDADE ENTRE PRODUTOS
# =========================================

# Transpor: produto × cliente
matriz_produto_usuario = matriz_usuario_produto.T

# Calcular cosine similarity
similaridade = cosine_similarity(matriz_produto_usuario)

# Transformar em DataFrame
similaridade_df = pd.DataFrame(
    similaridade,
    index=matriz_produto_usuario.index,
    columns=matriz_produto_usuario.index
)

print("\nMatriz de Similaridade:")
print(similaridade_df)



# =========================================
# 5. PRODUTO ALVO
# =========================================

produto_alvo_nome = "GPS Garmin Vortex Maré Drift"

# Buscar ID do produto
produto_alvo_id = produtos.loc[
    produtos['name'] == produto_alvo_nome, 'code'
].values[0]

print(f"\nProduto alvo: {produto_alvo_nome}")
print(f"ID: {produto_alvo_id}")


# =========================================
# 6. RANKING DE PRODUTOS SIMILARES
# =========================================

# Pegar similaridades do produto alvo
similaridades_produto = similaridade_df[produto_alvo_id]

# Remover o próprio produto
similaridades_produto = similaridades_produto.drop(produto_alvo_id)

# Ordenar e pegar top 5
top5_similares = similaridades_produto.sort_values(ascending=False).head(5)

# Transformar em DataFrame
top5_df = top5_similares.reset_index()
top5_df.columns = ['id_product', 'similaridade']

# Adicionar nomes dos produtos
top5_df = top5_df.merge(
    produtos[['code', 'name']],
    left_on='id_product',
    right_on='code',
    how='left'
)

top5_df = top5_df[['id_product', 'name', 'similaridade']]

print("\nTop 5 produtos similares:")
print(top5_df)


# =========================================
# 7. RECOMENDAÇÃO FINAL
# =========================================

produto_recomendado = top5_df.iloc[0]

print("\nProduto recomendado:")
print(produto_recomendado)
