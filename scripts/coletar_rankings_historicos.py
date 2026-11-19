"""Coleta sequencial explícita; reusa o parser fixado e conserva somente os JSONs e metadados reutilizados na coleta."""
import contextlib
import hashlib
import io
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from scrapy.http import HtmlResponse

from executar_piloto_hltv import normalize, EXPECTED_COMMIT

PROJECT = Path(r"C:\Users\roger.peloso\Desktop\major")
COLLECTOR = PROJECT.parent / "hltv-scraper-api"
sys.path.insert(0, str(COLLECTOR))
from hltv_scraper.hltv_scraper.spiders.hltv_top30 import HltvTop30Spider

MONTHS = "january february march april may june july august september october november december".split()
ROOT = PROJECT / "dados_externos" / "hltv" / "historico"


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    start, end = date(2023, 1, 9), date(2026, 11, 9)
    editions = [start + timedelta(days=7 * n) for n in range((end - start).days // 7 + 1)]
    session = requests.Session()
    session.headers["User-Agent"] = "hltv_scraper (+https://www.hltv.org/)"
    records, failures, index = [], [], []
    consecutive_failures = 0
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(f"Coleta de {len(editions)} edicoes semanais, uma requisicao por vez.", flush=True)
    for number, publication in enumerate(editions, 1):
        iso_date = publication.isoformat()
        folder = ROOT / iso_date
        folder.mkdir(exist_ok=True)
        url = f"https://www.hltv.org/ranking/teams/{publication.year}/{MONTHS[publication.month - 1]}/{publication.day}"
        metadata_path = folder / "metadados.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_bytes())
        try:
            if metadata.get("status") == "conteudo_validado":
                raw = json.loads((folder / "ranking_bruto.json").read_bytes())
            else:
                metadata = {
                    "data_solicitada": iso_date, "url_fonte": url,
                    "coletado_em_utc": datetime.now(timezone.utc).isoformat(),
                    "commit_coletor": EXPECTED_COMMIT, "adaptador_sha256": script_hash,
                    "transporte": "requests sequencial; parser original HltvTop30Spider",
                }
                response = session.get(url, timeout=(10, 40))
                metadata["http_status"] = response.status_code
                metadata["html_sha256"] = hashlib.sha256(response.content).hexdigest()
                response.raise_for_status()
                html = HtmlResponse(url=response.url, body=response.content, encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    raw = next(HltvTop30Spider(year=str(publication.year), month=MONTHS[publication.month - 1], day=publication.day).parse(html))
                save_json(folder / "ranking_bruto.json", raw)
            normalized = normalize(raw, publication=iso_date, source=url)
            if len(raw["ranking"]) != len(normalized):
                raise ValueError("A normalização alterou a quantidade de equipes.")
            players = {int(team["id"]): team["players"] for team in raw["ranking"]}
            for row in normalized:
                row["jogadores"] = players[row["hltv_team_id"]]
                row["coletado_em_utc"] = metadata["coletado_em_utc"]
                row["commit_coletor"] = EXPECTED_COMMIT
            metadata.update(status="conteudo_validado", data_recebida=raw["date"], equipes=len(normalized))
            records.extend(normalized)
            index.append(metadata)
            consecutive_failures = 0
        except Exception as error:
            consecutive_failures += 1
            metadata.update(status="falha", erro=f"{type(error).__name__}: {error}")
            failures.append(metadata)
            print("FALHA", iso_date, metadata["erro"], flush=True)
        if metadata.get("status") == "conteudo_validado":
            save_json(metadata_path, metadata)
        if number % 5 == 0 or number == len(editions):
            print(f"{number}/{len(editions)}: ate {iso_date}; {len(index)} validadas, {len(failures)} falhas.", flush=True)
        if consecutive_failures >= 3:
            print("Interrompendo apos tres falhas consecutivas; nao insistir no acesso.", flush=True)
            break
        time.sleep(3)
    save_json(ROOT / "rankings_historicos.json", records)
    summary = {"edicoes_planejadas": len(editions), "edicoes_validadas": len(index), "falhas": len(failures), "linhas": len(records)}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
