"""Piloto explícito: inicia somente a API local, consulta uma edição e arquiva evidências."""
import csv
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT = Path(r"C:\Users\roger.peloso\Desktop\major")
COLLECTOR = PROJECT.parent / "hltv-scraper-api"
PYTHON = COLLECTOR / ".venv" / "Scripts" / "python.exe"
EXPECTED_COMMIT = "a1102b131e4d6d2d7d10335e490aebd736248a59"
PUBLICATION = "2025-12-15"
PATH = "/api/v1/teams/rankings/hltv/2025/december/15"
URL = "http://127.0.0.1:8000" + PATH
SOURCE = "https://www.hltv.org/ranking/teams/2025/december/15"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize(payload, publication=PUBLICATION, source=SOURCE):
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"Esperada uma edição; recebidos {len(payload)} objetos.")
        payload = payload[0]
    if not isinstance(payload, dict) or not payload:
        raise ValueError("A API retornou um objeto vazio ou uma estrutura inválida.")
    if payload.get("date") != publication:
        raise ValueError(f"Data divergente: {payload.get('date')!r}.")
    if payload.get("ranking_type") != "hltv":
        raise ValueError("Tipo de ranking diferente de HLTV.")
    teams = payload.get("ranking")
    if not isinstance(teams, list) or not teams:
        raise ValueError("A publicação não contém equipes.")
    normalized = []
    for team in teams:
        if not isinstance(team, dict):
            raise ValueError("Registro de equipe inválido.")
        name = str(team.get("name") or "").strip()
        raw_points = str(team.get("points") or "").strip()
        points_match = re.fullmatch(r"\(?\s*(\d+)\s*(?:HLTV\s+)?points?\s*\)?", raw_points, re.I)
        if raw_points.isdecimal():
            points = int(raw_points)
        elif points_match:
            points = int(points_match.group(1))
        else:
            raise ValueError(f"Pontos não reconhecidos para {name!r}: {raw_points!r}.")
        raw_position = str(team.get("position") or "").strip().lstrip("#")
        raw_id = str(team.get("id") or "").strip()
        if not name or not raw_position.isdecimal() or not raw_id.isdecimal():
            raise ValueError(f"Identidade ou posição inválida: {team!r}.")
        position, team_id = int(raw_position), int(raw_id)
        if min(position, team_id) < 1:
            raise ValueError("Posição e ID precisam ser positivos.")
        link = str(team.get("link") or "")
        if not re.search(rf"/team/{team_id}(?:/|$)", link):
            raise ValueError(f"Link não confirma ID {team_id}: {link!r}.")
        normalized.append({
            "data_publicada": publication, "ranking_type": "hltv",
            "hltv_team_id": team_id, "nome_hltv": name,
            "posicao": position, "pontos": points, "url_fonte": source,
        })
    ids = [team["hltv_team_id"] for team in normalized]
    positions = [team["posicao"] for team in normalized]
    if len(set(ids)) != len(ids) or len(set(positions)) != len(positions):
        raise ValueError("IDs ou posições duplicados na edição.")
    normalized.sort(key=lambda team: team["posicao"])
    return normalized


def main():
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("Execute com o Python do ambiente do coletor.")
    commit = subprocess.check_output(["git", "-C", str(COLLECTOR), "rev-parse", "HEAD"], text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Commit inesperado: {commit}")
    model = PROJECT / "modelos" / "modelo_v1.joblib"
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    collected_at = datetime.now(timezone.utc)
    run_id = collected_at.strftime("%Y%m%dT%H%M%SZ")
    output = PROJECT / "dados_externos" / "hltv" / ("piloto_2025-12-15_" + run_id)
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PATH"] = str(PYTHON.parent) + os.pathsep + env["PATH"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["FLASK_DEBUG"] = "0"
    scrapy_path = shutil.which("scrapy", path=env["PATH"])
    if not scrapy_path or Path(scrapy_path).parent.resolve() != PYTHON.parent.resolve():
        raise RuntimeError("O scrapy do PATH não pertence ao ambiente do coletor.")
    metadata = {
        "coletado_em_utc": collected_at.isoformat(), "commit_coletor": commit,
        "url_api": URL, "url_fonte": SOURCE, "data_solicitada": PUBLICATION,
        "status": "iniciado", "modelo_v1_sha256": model_hash,
        "api_http_status": None, "equipes": 0,
    }
    write_json(output / "metadados.json", metadata)
    print("ARQUIVOS:", output, flush=True)
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            raise RuntimeError("A porta 8000 já está ocupada; não iniciar nem encerrar outro serviço.")
    process = None
    session = requests.Session()
    session.trust_env = False
    try:
        process = subprocess.Popen(
            [str(PYTHON), "-m", "flask", "--app", "app:flask_app", "run",
             "--host", "127.0.0.1", "--port", "8000", "--no-debugger", "--no-reload"],
            cwd=COLLECTOR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("O serviço encerrou antes de aceitar conexões; verifique a configuração do serviço.")
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", 8000)) == 0:
                    break
            time.sleep(0.1)
        else:
            raise RuntimeError("A API não iniciou no limite de 10 segundos.")
        print("API local iniciada. Consultando somente", PATH, flush=True)
        response = session.get(URL, timeout=(5, 90))
        metadata["api_http_status"] = response.status_code
        metadata["api_content_type"] = response.headers.get("content-type")
        metadata["resposta_sha256"] = hashlib.sha256(response.content).hexdigest()
        response.raise_for_status()
        records = normalize(response.json())
        with (output / "ranking_normalizado.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        metadata.update(status="conteudo_validado", data_recebida=PUBLICATION, equipes=len(records), amostra=records[:3])
    except Exception as error:
        metadata.update(status="falha", erro=f"{type(error).__name__}: {error}")
        print("FALHA:", metadata["erro"], flush=True)
    finally:
        if process is not None and process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                process.terminate()
            process.wait(timeout=15)
        session.close()
        if hashlib.sha256(model.read_bytes()).hexdigest() != model_hash:
            raise RuntimeError("O modelo V1 mudou durante o piloto.")
        write_json(output / "metadados.json", metadata)
        print(json.dumps(metadata, ensure_ascii=True, indent=2), flush=True)
    return output


if __name__ == "__main__":
    main()
