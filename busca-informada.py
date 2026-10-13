import heapq
import time
from dataclasses import dataclass, field
from itertools import count
from typing import Callable, Iterator

Estado = tuple[int, ...]

ESTADO_OBJETIVO: Estado = (
    1, 2, 3,
    4, 5, 6,
    7, 8, 0,
)

CUSTO_MOVIMENTO = 1

ACOES = {
    "CIMA": (-1, 0),
    "BAIXO": (1, 0),
    "ESQUERDA": (0, -1),
    "DIREITA": (0, 1),
}

INSTANCIA_A: Estado = (
    1, 5, 2,
    4, 8, 3,
    0, 7, 6,
)
INSTANCIA_B: Estado = (
    5, 8, 2,
    1, 7, 3,
    0, 4, 6,
)
INSTANCIA_C: Estado = (
    8, 7, 2,
    5, 4, 3,
    0, 1, 6,
)
INSTANCIAS: dict[str, Estado] = {
    "A": INSTANCIA_A,
    "B": INSTANCIA_B,
    "C": INSTANCIA_C,
}

Heuristica = Callable[[Estado], int]

@dataclass
class No:
    estado: Estado
    pai: "No | None" = None
    acao: str | None = None
    g: int = 0
    profundidade: int = 0

@dataclass
class MetricasBusca:
    solucao_encontrada: bool = False
    custo_solucao: int | None = None
    profundidade: int | None = None
    nos_expandidos: int = 0
    nos_gerados: int = 0
    fronteira_maxima: int = 0
    tempo_execucao: float = 0.0

@dataclass
class ResultadoBusca:
    nos_solucao: list[No] = field(default_factory=list)
    metricas: MetricasBusca = field(default_factory=MetricasBusca)

def validar_estado(estado: Estado) -> None:
    if len(estado) != 9 or any(type(peca) is not int for peca in estado):
        raise ValueError("O estado deve conter nove inteiros de 0 a 8.")

    if set(estado) != set(range(9)):
        raise ValueError("Cada valor de 0 a 8 deve aparecer exatamente uma vez.")

def gerar_sucessores(estado: Estado) -> Iterator[tuple[str, Estado, int]]:
    validar_estado(estado)
    indice_vazio = estado.index(0)
    linha, coluna = divmod(indice_vazio, 3)

    for acao, (variacao_linha, variacao_coluna) in ACOES.items():
        nova_linha = linha + variacao_linha
        nova_coluna = coluna + variacao_coluna

        if 0 <= nova_linha < 3 and 0 <= nova_coluna < 3:
            novo_indice = nova_linha * 3 + nova_coluna
            pecas = list(estado)
            pecas[indice_vazio], pecas[novo_indice] = (
                pecas[novo_indice],
                pecas[indice_vazio],
            )
            yield acao, tuple(pecas), CUSTO_MOVIMENTO

def h1(estado: Estado) -> int:
    validar_estado(estado)
    return sum(
        peca != 0 and peca != ESTADO_OBJETIVO[indice]
        for indice, peca in enumerate(estado)
    )

def h2(estado: Estado) -> int:
    validar_estado(estado)
    distancia = 0

    for indice, peca in enumerate(estado):
        if peca == 0:
            continue

        linha, coluna = divmod(indice, 3)
        linha_objetivo, coluna_objetivo = divmod(peca - 1, 3)
        distancia += abs(linha - linha_objetivo) + abs(coluna - coluna_objetivo)

    return distancia

def reconstruir_caminho(no_objetivo: No) -> list[No]:
    caminho: list[No] = []
    atual: No | None = no_objetivo

    while atual is not None:
        caminho.append(atual)
        atual = atual.pai

    return list(reversed(caminho))

def busca_gulosa(
    estado_inicial: Estado,
    heuristica: Heuristica,
) -> ResultadoBusca:
    inicio_tempo = time.perf_counter()
    validar_estado(estado_inicial)

    metricas = MetricasBusca()
    visitados: set[Estado] = set()
    fila_prioridade: list[tuple[int, int, No]] = []
    contador = count()
    no_inicial = No(estado=estado_inicial)

    heapq.heappush(
        fila_prioridade,
        (heuristica(estado_inicial), next(contador), no_inicial),
    )
    metricas.fronteira_maxima = 1

    while fila_prioridade:
        _, _, no_atual = heapq.heappop(fila_prioridade)

        if no_atual.estado == ESTADO_OBJETIVO:
            metricas.solucao_encontrada = True
            metricas.custo_solucao = no_atual.g
            metricas.profundidade = no_atual.profundidade
            caminho = reconstruir_caminho(no_atual)
            metricas.tempo_execucao = time.perf_counter() - inicio_tempo
            return ResultadoBusca(nos_solucao=caminho, metricas=metricas)

        if no_atual.estado in visitados:
            continue

        visitados.add(no_atual.estado)
        metricas.nos_expandidos += 1

        for acao, estado_sucessor, custo in gerar_sucessores(no_atual.estado):
            metricas.nos_gerados += 1

            if estado_sucessor in visitados:
                continue

            no_filho = No(
                estado=estado_sucessor,
                pai=no_atual,
                acao=acao,
                g=no_atual.g + custo,
                profundidade=no_atual.profundidade + 1,
            )

            heapq.heappush(
                fila_prioridade,
                (heuristica(estado_sucessor), next(contador), no_filho),
            )
            metricas.fronteira_maxima = max(
                metricas.fronteira_maxima,
                len(fila_prioridade),
            )

    metricas.tempo_execucao = time.perf_counter() - inicio_tempo
    return ResultadoBusca(metricas=metricas)

def busca_a_estrela(
    estado_inicial: Estado,
    heuristica: Heuristica,
) -> ResultadoBusca:
    inicio_tempo = time.perf_counter()
    validar_estado(estado_inicial)

    metricas = MetricasBusca()
    no_inicial = No(estado=estado_inicial)
    melhor_g: dict[Estado, int] = {estado_inicial: 0}
    fechados: set[Estado] = set()
    fronteira: list[tuple[int, int, No]] = []
    contador = count()

    heapq.heappush(
        fronteira,
        (
            no_inicial.g + heuristica(estado_inicial),
            next(contador),
            no_inicial,
        ),
    )
    metricas.fronteira_maxima = 1

    while fronteira:
        _, _, no_atual = heapq.heappop(fronteira)

        if no_atual.g != melhor_g[no_atual.estado]:
            continue

        if no_atual.estado in fechados:
            continue

        if no_atual.estado == ESTADO_OBJETIVO:
            metricas.solucao_encontrada = True
            metricas.custo_solucao = no_atual.g
            metricas.profundidade = no_atual.profundidade
            caminho = reconstruir_caminho(no_atual)
            metricas.tempo_execucao = time.perf_counter() - inicio_tempo
            return ResultadoBusca(nos_solucao=caminho, metricas=metricas)

        fechados.add(no_atual.estado)
        metricas.nos_expandidos += 1

        for acao, estado_sucessor, custo in gerar_sucessores(no_atual.estado):
            metricas.nos_gerados += 1
            g_provisorio = no_atual.g + custo

            if g_provisorio >= melhor_g.get(estado_sucessor, float("inf")):
                continue

            melhor_g[estado_sucessor] = g_provisorio
            fechados.discard(estado_sucessor)

            no_filho = No(
                estado=estado_sucessor,
                pai=no_atual,
                acao=acao,
                g=g_provisorio,
                profundidade=no_atual.profundidade + 1,
            )

            h = heuristica(estado_sucessor)
            f = g_provisorio + h
            heapq.heappush(fronteira, (f, next(contador), no_filho))
            metricas.fronteira_maxima = max(
                metricas.fronteira_maxima,
                len(fronteira),
            )

    metricas.tempo_execucao = time.perf_counter() - inicio_tempo
    return ResultadoBusca(metricas=metricas)

def imprimir_caminho(nos_solucao: list[No]) -> None:
    for indice, no in enumerate(nos_solucao):
        if indice:
            print()

        print(f"Estado {indice}")

        for inicio in range(0, 9, 3):
            print(
                " ".join(
                    "_" if peca == 0 else str(peca)
                    for peca in no.estado[inicio:inicio + 3]
                )
            )

def imprimir_metricas(metricas: MetricasBusca) -> None:
    print(f"Solução encontrada: {'Sim' if metricas.solucao_encontrada else 'Não'}")
    print(
        f"Custo da solução g(G): "
        f"{metricas.custo_solucao if metricas.custo_solucao is not None else 'Não se aplica'}"
    )
    print(
        f"Profundidade: "
        f"{metricas.profundidade if metricas.profundidade is not None else 'Não se aplica'}"
    )
    print(f"Nós expandidos: {metricas.nos_expandidos}")
    print(f"Nós gerados: {metricas.nos_gerados}")
    print(f"Fronteira máxima: {metricas.fronteira_maxima}")
    print(f"Tempo total: {metricas.tempo_execucao:.6f} s")

def principal() -> None:
    opcoes = {
        "1": (busca_gulosa, h1),
        "2": (busca_gulosa, h2),
        "3": (busca_a_estrela, h1),
        "4": (busca_a_estrela, h2),
    }

    try:
        instancia = input(
            "Instância: A (fácil), B (intermediária) ou C (difícil): "
        ).strip().upper()

        while instancia not in INSTANCIAS:
            instancia = input(
                "Instância inválida.\nDigite A, B ou C: "
            ).strip().upper()

        print(
            "1. Greedy + h1\n"
            "2. Greedy + h2\n"
            "3. A* + h1\n"
            "4. A* + h2"
        )

        escolha = input("Escolha uma opção (1 a 4): ").strip()

        while escolha not in opcoes:
            escolha = input(
                "Opção inválida.\nDigite 1, 2, 3 ou 4: "
            ).strip()

        busca, heuristica = opcoes[escolha]
        resultado = busca(INSTANCIAS[instancia], heuristica)

    except (EOFError, KeyboardInterrupt):
        print("\nExecução encerrada.")
        return

    imprimir_caminho(resultado.nos_solucao)
    imprimir_metricas(resultado.metricas)

if __name__ == "__main__":
    principal()