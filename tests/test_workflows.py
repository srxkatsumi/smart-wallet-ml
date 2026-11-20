"""
Os workflows precisam ser YAML que o GitHub aceita. O PyYAML padrão aceita chave
duplicada em silêncio (fica a última), mas o GitHub recusa o arquivo inteiro e
deixa de disparar os agendamentos: foi o que aconteceu em 19/11/2026, com dois
blocos `env:` no mesmo passo, e a execução da noite não rodou.
"""
from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).resolve().parent.parent / ".github" / "workflows").glob("*.yml"))


class _SemDuplicadas(yaml.SafeLoader):
    pass


def _mapa(loader, node, deep=False):
    vistos = set()
    for k, _ in node.value:
        chave = loader.construct_object(k, deep=deep)
        if chave in vistos:
            raise ValueError(f"chave duplicada '{chave}' na linha {k.start_mark.line + 1}")
        vistos.add(chave)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_SemDuplicadas.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapa)


@pytest.mark.parametrize("arquivo", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_sem_chaves_duplicadas(arquivo):
    yaml.load(arquivo.read_text(encoding="utf-8"), Loader=_SemDuplicadas)


def test_teste_pega_o_erro_de_19_11():
    with pytest.raises(ValueError, match="env"):
        yaml.load("step:\n  env:\n    A: 1\n  env:\n    B: 2\n", Loader=_SemDuplicadas)
