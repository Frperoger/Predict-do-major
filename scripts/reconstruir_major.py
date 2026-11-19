"""Reconstrução reproduzível da base, identidade histórica, mapas e validação."""
import argparse
import bisect
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from modelo_simetrico import ModeloSimetrico

PROJECT = Path(r"C:\Users\roger.peloso\Desktop\major")
OUTPUT = PROJECT / "dados_externos" / "reconstrucao"
V1_COLUMNS = ["dif_jogos_anteriores", "dif_vitorias_anteriores", "dif_taxa_vitorias_anteriores", "bestOf"]


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_v1():
    raw = pd.read_csv(PROJECT / "cs2_tier1_games.csv")
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    duplicates = raw["match_id"].isin([7298940, 7287899]) & raw["is_total"].eq(True) & raw["tournament"].eq("PGL Astana 2026. Closed European Qualifier")
    clean = raw.loc[~duplicates].copy()
    series = clean.loc[clean["is_total"].eq(True)].copy().sort_values("datetime")
    if series["match_id"].duplicated().any():
        raise ValueError("Totais duplicados não previstos na regra original.")
    series["team1_win_corrigido"] = np.where(series["score1_match"] > series["score2_match"], 1, np.where(series["score1_match"] < series["score2_match"], 0, np.nan))
    series = series.loc[series["team1_win_corrigido"].notna()].copy()
    series["team1_win_corrigido"] = series["team1_win_corrigido"].astype("int64")
    first = series[["match_id", "datetime", "team1", "team2", "team1_win_corrigido"]].rename(columns={"team1": "equipe", "team2": "adversario", "team1_win_corrigido": "venceu"})
    second = series[["match_id", "datetime", "team1", "team2", "team1_win_corrigido"]].rename(columns={"team2": "equipe", "team1": "adversario", "team1_win_corrigido": "venceu"})
    second["venceu"] = 1 - second["venceu"]
    history = pd.concat([first, second], ignore_index=True).sort_values("datetime")
    blocks = history.groupby(["equipe", "datetime"]).agg(jogos_no_horario=("match_id", "size"), vitorias_no_horario=("venceu", "sum")).reset_index().sort_values(["equipe", "datetime"])
    blocks["jogos_anteriores"] = blocks.groupby("equipe")["jogos_no_horario"].cumsum() - blocks["jogos_no_horario"]
    blocks["vitorias_anteriores"] = blocks.groupby("equipe")["vitorias_no_horario"].cumsum() - blocks["vitorias_no_horario"]
    blocks["taxa_vitorias_anteriores"] = blocks["vitorias_anteriores"] / blocks["jogos_anteriores"]
    columns = ["jogos_anteriores", "vitorias_anteriores", "taxa_vitorias_anteriores"]
    for side in [1, 2]:
        attributes = blocks[["equipe", "datetime"] + columns].rename(columns={"equipe": f"team{side}", **{column: f"team{side}_{column}" for column in columns}})
        series = series.merge(attributes, on=[f"team{side}", "datetime"], how="left", validate="many_to_one")
    for column in columns:
        series["dif_" + column] = series["team1_" + column] - series["team2_" + column]
    if len(series) != 5190:
        raise ValueError(f"Carga diferente da auditada: {len(series)} séries; revisar antes de comparar.")
    return raw, series


def prepare_maps(raw, series):
    maps = raw.loc[raw["is_total"].eq(False)].copy()
    maps["motivo_rejeicao"] = ""
    no_name = maps["map_name"].fillna("").str.strip().eq("")
    no_score = maps[["score1_game", "score2_game"]].isna().any(axis=1)
    tie = maps["score1_game"].eq(maps["score2_game"])
    maps.loc[no_name, "motivo_rejeicao"] += "sem_nome;"
    maps.loc[no_score, "motivo_rejeicao"] += "sem_placar;"
    maps.loc[tie, "motivo_rejeicao"] += "empate;"
    rejected = maps.loc[maps["motivo_rejeicao"].ne("")].copy()
    maps = maps.loc[maps["motivo_rejeicao"].eq("")].copy().drop(columns="motivo_rejeicao")
    for key in [["match_id", "game_id"], ["match_id", "map_name"]]:
        if maps.duplicated(key).any():
            raise ValueError(f"Mapas duplicados por {key}; investigar antes de acumular.")
    maps["team1_win_mapa"] = maps["score1_game"].gt(maps["score2_game"]).astype("int64")
    map_columns = ["match_id", "game_id", "datetime", "team1", "team2", "map_name", "score1_game", "score2_game", "team1_win_mapa"]
    maps[map_columns].to_csv(OUTPUT / "df_mapas.csv", index=False)
    rejected[["match_id", "game_id", "datetime", "team1", "team2", "map_name", "score1_game", "score2_game", "motivo_rejeicao"]].to_csv(OUTPUT / "mapas_rejeitados.csv", index=False)

    # O vocabulário de mapas é aprendido somente no treino, sem presumir o pool vigente.
    vocabulary = sorted(maps.loc[maps["datetime"].lt("2026-01-01"), "map_name"].unique())
    history = []
    for side in [1, 2]:
        view = maps[["match_id", "game_id", "datetime", "map_name", f"team{side}", "team1_win_mapa"]].rename(columns={f"team{side}": "equipe", "team1_win_mapa": "venceu"})
        if side == 2:
            view["venceu"] = 1 - view["venceu"]
        history.append(view)
    history = pd.concat(history, ignore_index=True)
    blocks = history.groupby(["equipe", "map_name", "datetime"]).agg(jogos_bloco=("game_id", "size"), vitorias_bloco=("venceu", "sum")).reset_index().sort_values(["equipe", "map_name", "datetime"])
    blocks["jogos_ate_horario"] = blocks.groupby(["equipe", "map_name"])["jogos_bloco"].cumsum()
    blocks["vitorias_ate_horario"] = blocks.groupby(["equipe", "map_name"])["vitorias_bloco"].cumsum()

    perspectives = []
    for side in [1, 2]:
        view = series[["match_id", "datetime", f"team{side}"]].rename(columns={f"team{side}": "equipe"})
        view["lado"] = side
        perspectives.append(view)
    left = pd.concat(perspectives, ignore_index=True).sort_values("datetime")
    profiles, features = [], series[["match_id"]].copy()
    feature_columns = []
    all_maps = sorted(maps['map_name'].unique())
    for map_name in all_maps:
        right = blocks.loc[blocks["map_name"].eq(map_name), ["equipe", "datetime", "jogos_ate_horario", "vitorias_ate_horario"]].rename(columns={"datetime": "data_ultimo_mapa"}).sort_values("data_ultimo_mapa")
        joined = pd.merge_asof(left, right, left_on="datetime", right_on="data_ultimo_mapa", by="equipe", direction="backward", allow_exact_matches=False)
        if not (joined["data_ultimo_mapa"].isna() | joined["data_ultimo_mapa"].lt(joined["datetime"])).all():
            raise ValueError("Histórico de mapa contém o horário atual ou posterior.")
        joined["map_name"] = map_name
        joined["jogos_anteriores"] = joined["jogos_ate_horario"].fillna(0).astype("int64")
        joined["vitorias_anteriores"] = joined["vitorias_ate_horario"].fillna(0).astype("int64")
        joined["taxa_vitorias_anteriores"] = joined["vitorias_anteriores"] / joined["jogos_anteriores"].where(joined["jogos_anteriores"].gt(0))
        profiles.append(joined[["match_id", "datetime", "equipe", "lado", "map_name", "data_ultimo_mapa", "jogos_anteriores", "vitorias_anteriores", "taxa_vitorias_anteriores"]])
        if map_name not in vocabulary:
            continue
        pivot = joined.pivot(index="match_id", columns="lado", values=["jogos_anteriores", "taxa_vitorias_anteriores"])
        for attribute in ["jogos_anteriores", "taxa_vitorias_anteriores"]:
            name = f"dif_{attribute}_{map_name.lower()}"
            delta = (pivot[(attribute, 1)] - pivot[(attribute, 2)]).rename(name).reset_index()
            features = features.merge(delta, on="match_id", validate="one_to_one")
            feature_columns.append(name)
    profile = pd.concat(profiles, ignore_index=True)
    profile.to_csv(OUTPUT / "df_historico_mapas.csv", index=False)
    features.to_csv(OUTPUT / "atributos_mapas.csv", index=False)

    # Auditoria independente do asof em exemplos concretos, inclusive sem histórico.
    checks = profile.sample(n=min(12, len(profile)), random_state=25)
    manual = []
    for row in checks.itertuples():
        past = history.loc[history["equipe"].eq(row.equipe) & history["map_name"].eq(row.map_name) & history["datetime"].lt(row.datetime)]
        if len(past) != row.jogos_anteriores or int(past["venceu"].sum()) != row.vitorias_anteriores:
            raise ValueError("A soma direta dos mapas divergiu do histórico temporal.")
        manual.append({"match_id": row.match_id, "equipe": row.equipe, "map_name": row.map_name, "jogos_anteriores": len(past), "vitorias_anteriores": int(past["venceu"].sum()), "conferido": True})
    save_json(OUTPUT / "conferencia_manual_mapas.json", manual)
    summary = {"linhas_mapas_brutas": int(raw["is_total"].eq(False).sum()), "mapas_utilizaveis": len(maps), "mapas_rejeitados": len(rejected), "vocabulario_aprendido_no_treino": vocabulary, "mapas_observados_na_base": all_maps, "linhas_perfis": len(profile), "atributos": feature_columns, "exemplos_conferidos_por_soma_direta": len(manual), "observacao": "Históricos por mapa antes do confronto; não representam o pool vigente nem uma previsão após o veto. Cache é auditado, mas não entra nas entradas porque apareceu somente após o treino."}
    save_json(OUTPUT / "auditoria_mapas.json", summary)
    return features, feature_columns, summary


def canonical_name(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    words = re.findall(r"[a-z0-9]+", text)
    return "".join(word for word in words if word not in {"team", "gaming", "esports", "esport", "clan"})


def nickname(value):
    if pd.isna(value):
        return None
    return re.split(r"\s*\(", str(value), maxsplit=1)[0].strip().casefold()


def ranking_features(series):
    history_path = PROJECT / "dados_externos" / "hltv" / "historico" / "rankings_historicos.json"
    ranks = json.loads(history_path.read_bytes())
    if not ranks:
        raise ValueError("Nenhuma edição histórica validada.")
    table = pd.DataFrame(ranks)
    if table.duplicated(["ranking_type", "data_publicada", "hltv_team_id"]).any():
        raise ValueError("Duplicatas no histórico de ranking.")
    editions = sorted(table["data_publicada"].unique())
    per_edition = {}
    for edition, group in table.groupby("data_publicada"):
        candidates = {}
        for record in group.to_dict("records"):
            name_key = canonical_name(record["nome_hltv"])
            if name_key:
                candidates.setdefault(name_key, []).append(record)
        per_edition[edition] = candidates
    mappings = []
    for game in series.to_dict("records"):
        day = game["datetime"].date().isoformat()
        index = bisect.bisect_left(editions, day) - 1
        edition = editions[index] if index >= 0 else None
        age = (game["datetime"].normalize() - pd.Timestamp(edition)).days if edition else None
        for side in [1, 2]:
            name = game[f"team{side}"]
            roster = {nick for i in range(1, 6) if (nick := nickname(game.get(f"team{side}_player{i}")))}
            record = {"match_id": game["match_id"], "datetime": game["datetime"], "lado": side, "nome_local": name, "validade_inicio": game["datetime"], "validade_fim": game["datetime"], "data_evidencia": edition, "dias_desde_ranking": age, "status": "pendente", "motivo": "", "hltv_team_id": np.nan, "nome_hltv": None, "pontos": np.nan, "posicao": np.nan, "jogadores_em_comum": 0, "url_evidencia": None}
            if name in {"1win Team", "1w Team"}:
                record["motivo"] = "identidade_reservada_para_investigacao"
            elif edition is None or age > 14:
                record["motivo"] = "edicao_anterior_ausente_ou_mais_de_14_dias"
            elif len(roster) < 3:
                record["motivo"] = "escalacao_local_insuficiente"
            else:
                candidates = per_edition[edition].get(canonical_name(name), [])
                corroborated = []
                for candidate in candidates:
                    overlap = roster & {str(player).casefold() for player in candidate["jogadores"]}
                    if len(overlap) >= 3:
                        corroborated.append((candidate, sorted(overlap)))
                if len(corroborated) == 1:
                    candidate, common = corroborated[0]
                    record.update(status="confirmada_no_confronto", motivo="nome_compativel_e_tres_ou_mais_jogadores_da_edicao_anterior", hltv_team_id=candidate["hltv_team_id"], nome_hltv=candidate["nome_hltv"], pontos=candidate["pontos"], posicao=candidate["posicao"], jogadores_em_comum=len(common), url_evidencia=candidate["url_fonte"], evidencia_jogadores=";".join(common))
                elif len(corroborated) > 1:
                    record["motivo"] = "mais_de_uma_identidade_compativel"
                elif not candidates:
                    record["motivo"] = "nome_sem_candidato_na_edicao"
                else:
                    record["motivo"] = "escalacao_nao_corrobora_nome"
            mappings.append(record)
    mapping = pd.DataFrame(mappings)
    if mapping.duplicated(["match_id", "lado"]).any():
        raise ValueError("Mais de uma equivalência por participação.")
    confirmed = mapping["status"].eq("confirmada_no_confronto")
    if not pd.to_datetime(mapping.loc[confirmed, "data_evidencia"]).lt(mapping.loc[confirmed, "datetime"].dt.normalize()).all():
        raise ValueError("Ranking do mesmo dia ou posterior associado.")
    mapping.to_csv(OUTPUT / "df_equivalencias_hltv.csv", index=False)
    pivot = mapping.pivot(index="match_id", columns="lado", values=["pontos", "posicao"])
    features = pd.DataFrame({"match_id": pivot.index, "dif_pontos_hltv": pivot[("pontos", 1)] - pivot[("pontos", 2)], "dif_posicao_hltv": pivot[("posicao", 2)] - pivot[("posicao", 1)]}).reset_index(drop=True)
    summary = {"edicoes": len(editions), "primeira_edicao": editions[0], "ultima_edicao": editions[-1], "linhas_rankings": len(table), "participacoes": len(mapping), "participacoes_confirmadas": int(confirmed.sum()), "percentual_confirmado": float(confirmed.mean() * 100), "motivos_pendentes": mapping.loc[~confirmed, "motivo"].value_counts().to_dict(), "idade_maxima_dias": 14, "escopo_validade": "Somente a participação conferida; sem extrapolar a identidade para outros períodos.", "limite": "A escalação local foi usada para corroborar a identidade, nunca o resultado; previsão real requer identidade/escalação disponível antes do confronto."}
    table.drop(columns="jogadores").to_csv(OUTPUT / "df_rank_historico.csv", index=False)
    return features, summary


def metrics(model, x, y):
    positive = list(model.classes_).index(1)
    probabilities = model.predict_proba(x)[:, positive]
    return {"accuracy": float(accuracy_score(y, model.predict(x))), "roc_auc": float(roc_auc_score(y, probabilities)), "log_loss": float(log_loss(y, probabilities)), "brier": float(brier_score_loss(y, probabilities))}


def new_pipeline():
    return Pipeline([("imputacao", SimpleImputer(strategy="median", add_indicator=True)), ("padronizacao", StandardScaler()), ("classificador", LogisticRegression(max_iter=1000))])


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_hash = hash_file(PROJECT / "modelos" / "modelo_v1.joblib")
    raw, series = base_v1()
    _, _, map_summary = prepare_maps(raw, series)
    v1 = joblib.load(PROJECT / "modelos" / "modelo_v1.joblib")
    valid = series["datetime"].ge("2026-01-01") & series["datetime"].lt("2026-07-01")
    result = metrics(v1, series.loc[valid, V1_COLUMNS], series.loc[valid, "team1_win_corrigido"])
    if not np.isclose(result["log_loss"], 0.6950171753, atol=1e-9):
        raise ValueError(f"A reprodução da V1 divergiu: {result}")
    if hash_file(PROJECT / "modelos" / "modelo_v1.joblib") != model_hash:
        raise RuntimeError("O modelo original mudou.")
    print(json.dumps({"series": len(series), "v1_validacao": result, "mapas": map_summary}, ensure_ascii=True), flush=True)


def finish():
    raw, series = base_v1()
    ranking, rank_summary = ranking_features(series)
    maps = pd.read_csv(OUTPUT / "atributos_mapas.csv")
    map_columns = [column for column in maps.columns if column != "match_id"]
    series = series.merge(ranking, on="match_id", how="left", validate="one_to_one").merge(maps, on="match_id", how="left", validate="one_to_one")
    train = series["datetime"].lt("2026-01-01")
    valid = series["datetime"].ge("2026-01-01") & series["datetime"].lt("2026-07-01")
    test = series["datetime"].ge("2026-07-01")
    if (int(train.sum()), int(valid.sum()), int(test.sum())) != (3775, 642, 773):
        raise ValueError("Os conjuntos diferem da V1.")
    coverage = []
    for name, mask in [("treino", train), ("validacao", valid), ("teste", test)]:
        coverage.append({"conjunto": name, "linhas": int(mask.sum()), "ranking_dois_lados": int(series.loc[mask, "dif_pontos_hltv"].notna().sum()), "percentual_ranking_dois_lados": float(series.loc[mask, "dif_pontos_hltv"].notna().mean() * 100)})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = PROJECT / "experimentos" / ("reconstrucao_" + run_id)
    experiment.mkdir(parents=True, exist_ok=False)
    definitions = {"V1": V1_COLUMNS, "V1 + ranking": V1_COLUMNS + ["dif_pontos_hltv"], "V1 + mapas": V1_COLUMNS + map_columns, "V1 + ranking + mapas": V1_COLUMNS + ["dif_pontos_hltv"] + map_columns}
    models, results = {}, []
    for name, columns in definitions.items():
        model = joblib.load(PROJECT / "modelos" / "modelo_v1.joblib") if name == "V1" else new_pipeline().fit(series.loc[train, columns], series.loc[train, "team1_win_corrigido"])
        models[name] = model
        result = {"modelo": name, "entradas": len(columns), "linhas_validacao": int(valid.sum()), **metrics(model, series.loc[valid, columns], series.loc[valid, "team1_win_corrigido"])}
        results.append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
    dummy = DummyClassifier(strategy="prior").fit(series.loc[train, V1_COLUMNS], series.loc[train, "team1_win_corrigido"])
    models["Referencia"] = dummy
    definitions["Referencia"] = V1_COLUMNS
    results.append({"modelo": "Referencia", "entradas": 0, "linhas_validacao": int(valid.sum()), **metrics(dummy, series.loc[valid, V1_COLUMNS], series.loc[valid, "team1_win_corrigido"])})
    eligible = []
    for name in list(definitions):
        symmetric_name = name + " (simetrico)"
        symmetric_model = ModeloSimetrico(models[name])
        columns = definitions[name]
        models[symmetric_name] = symmetric_model
        definitions[symmetric_name] = columns
        result = {"modelo": symmetric_name, "entradas": len(columns), "linhas_validacao": int(valid.sum()), **metrics(symmetric_model, series.loc[valid, columns], series.loc[valid, "team1_win_corrigido"])}
        results.append(result)
        eligible.append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
    series[["match_id", "datetime", "team1", "team2", "team1_win_corrigido"] + list(dict.fromkeys(V1_COLUMNS + ["dif_pontos_hltv", "dif_posicao_hltv"] + map_columns))].to_csv(experiment / "base_experimentos.csv", index=False)
    # Para o torneio, a chance de A vencer B precisa complementar a de B vencer A.
    # A simetrização é comparada na validação antes de qualquer uso do teste.
    chosen = min(eligible, key=lambda row: (row["log_loss"], row["brier"]))
    selection = {"modelo": chosen["modelo"], "regra": "Entre versões simétricas, menor Log Loss na validação; Brier desempata. A referência simétrica (50%) também participa. Versões originais são referências de comparação.", "entradas": definitions[chosen["modelo"]], "ajuste": "Parâmetros e transformações aprendidos somente no treino; sem retreino após seleção.", "fixado_em_utc": datetime.now(timezone.utc).isoformat(), "teste_avaliado_neste_momento": False, "hash_csv": hash_file(PROJECT / "cs2_tier1_games.csv"), "hash_modelo_v1": hash_file(PROJECT / "modelos" / "modelo_v1.joblib"), "regras": {"ranking": "Edição estritamente anterior ao dia; idade máxima 14 dias; ausência não vira rank zero.", "identidade": "Nome compatível e pelo menos três jogadores em comum com a edição anterior; validade restrita à participação.", "mapas": "Somente histórico anterior ao horário; vocabulário aprendido no treino, sem usar veto/mapas do confronto atual.", "ordem_equipes": "Média de p(A,B) e 1-p(B,A); bestOf mantido e todas as diferenças invertidas."}}
    save_json(experiment / "selecao_antes_do_teste.json", selection)
    selected_model = models[chosen["modelo"]]
    joblib.dump(selected_model, experiment / "modelo_selecionado.joblib", compress=3)
    test_results = []
    for name in dict.fromkeys([chosen["modelo"], "V1", "Referencia"]):
        test_results.append({"modelo": name, "linhas_teste": int(test.sum()), **metrics(models[name], series.loc[test, definitions[name]], series.loc[test, "team1_win_corrigido"])})
    report = {"pasta_experimento": str(experiment), "metricas_validacao": results, "melhor_log_loss_validacao": chosen["modelo"], "teste_avaliado": True, "metricas_teste_final": test_results, "selecao_previa_sha256": hash_file(experiment / "selecao_antes_do_teste.json"), "colunas_por_modelo": definitions, "cobertura_rankings": coverage, "auditoria_rankings": rank_summary, "hash_csv": hash_file(PROJECT / "cs2_tier1_games.csv"), "hash_modelo_v1": hash_file(PROJECT / "modelos" / "modelo_v1.joblib"), "limite_do_teste": "Avaliação única após fixar a versão. Nenhuma escolha foi alterada com os resultados do teste; novas escolhas exigirão nova evidência fora desta amostra."}
    save_json(OUTPUT / "ultimo_experimento.json", report)
    print(json.dumps({"experimento": str(experiment), "melhor_validacao": chosen["modelo"], "teste_avaliado": True, "metricas_teste_final": test_results, "cobertura": coverage}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("etapa", choices=["preparar", "concluir"])
    arguments = parser.parse_args()
    {"preparar": prepare, "concluir": finish}[arguments.etapa]()
