"""
Saúde do sistema para o topo do email: dá para confiar nas previsões de hoje?

Checa, para os ativos da carteira:
  - se há previsão para cada um;
  - se a última barra é a do último pregão encerrado no calendário do ativo
    (uma barra velha quer dizer previsão feita com dados de ontem);
  - quais etapas do pipeline falharam (as que "não bloqueiam" e só iam para o log);
  - avisos do contexto (VIX/SPY preenchidos para a frente).

Estado: "ok", "atencao" (uma etapa secundária falhou, as previsões da carteira
continuam válidas) ou "problema" (dados da carteira ausentes ou velhos: as
previsões desses ativos não são confiáveis). Etapas que travam a produção
(download, treino) derrubam a execução inteira e nem geram email.
"""
import pandas as pd

from data.calendars import last_completed_session


def avaliar_saude(carteira: list[str], resultados_ml: dict, falhas: list[tuple[str, str]],
                  context_warnings: list[str] | None = None,
                  now: pd.Timestamp | None = None) -> dict:
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    sem_previsao, desatualizados = [], []
    for t in carteira:
        res = resultados_ml.get(t)
        if not res or "last_date" not in res:
            sem_previsao.append(t)
            continue
        esperado = last_completed_session(t, now)
        ultima = pd.Timestamp(res["last_date"]).normalize()
        if esperado is not None and ultima < esperado:
            desatualizados.append((t, ultima.strftime("%d/%m"), esperado.strftime("%d/%m")))

    problemas, atencoes = [], []
    if sem_previsao:
        problemas.append(f"Sem previsão hoje: {', '.join(sem_previsao)}")
    for t, ultima, esperado in desatualizados:
        problemas.append(f"{t}: última barra de {ultima}, esperado {esperado} (previsão com dados velhos)")
    for etapa, erro in falhas:
        atencoes.append(f"Etapa '{etapa}' falhou: {erro[:120]}")
    for w in context_warnings or []:
        atencoes.append(w)

    estado = "problema" if problemas else ("atencao" if atencoes else "ok")
    return {
        "estado": estado,
        "problemas": problemas,
        "atencoes": atencoes,
        "n_ativos": len(carteira),
        "n_ok": len(carteira) - len(sem_previsao) - len(desatualizados),
    }
