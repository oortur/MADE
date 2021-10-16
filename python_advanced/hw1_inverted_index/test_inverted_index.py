from textwrap import dedent
from argparse import Namespace
import pytest

from task_Bulanbaev_Artur_inverted_index import *


def test_can_load_documents_v1():
    documents = load_documents(DEFAULT_DATASET_PATH)
    etalon_documents = {
        "13": "If we go again all the way from the start",
        "666": "On and on the rain will fall , like tears from the star",
        "7": "Don't fall on my faith , oh",
        "123": "You were the one thing in my way"
    }
    assert etalon_documents == documents, (
        "load_documents loaded dataset incorrectly"
    )


def test_can_load_documents_v2(tmpdir):
    dataset_str = dedent("""\
        13	If we go again all the way from the start
        666	On and on the rain will fall , like tears from the star
        7	Don't fall on my faith , oh
        123	You were the one thing in my way
    """)
    dataset_fio = tmpdir.join("./temp.data")
    dataset_fio.write(dataset_str)
    documents = load_documents(dataset_fio)
    etalon_documents = {
        "13": "If we go again all the way from the start",
        "666": "On and on the rain will fall , like tears from the star",
        "7": "Don't fall on my faith , oh",
        "123": "You were the one thing in my way"
    }
    assert etalon_documents == documents, (
        "load_documents loaded dataset incorrectly"
    )


DATASET_TINY_STR = dedent("""\
    13	If we go again all the way from the start
    666	On and on the rain will fall , like tears from the star
    7	Don't fall on my faith , oh
    123	You were the one thing in my way
""")


@pytest.fixture()
def tiny_dataset_fio(tmpdir):
    dataset_fio = tmpdir.join("dataset.txt")
    dataset_fio.write(DATASET_TINY_STR)
    return dataset_fio


def test_can_load_documents_using_tmpdir(tiny_dataset_fio):
    documents = load_documents(tiny_dataset_fio)
    etalon_documents = {
        "13": "If we go again all the way from the start",
        "666": "On and on the rain will fall , like tears from the star",
        "7": "Don't fall on my faith , oh",
        "123": "You were the one thing in my way"
    }
    assert etalon_documents == documents, (
        "load_documents loaded dataset incorrectly"
    )


@pytest.mark.parametrize(
    "query, etalon_answer",
    [
        pytest.param(["fall"], ["666", "7"], id="fall"),
        pytest.param(["from"], ["666", "13"], id="from"),
        pytest.param(["fall", "from"], ["666"], id="two words"),
        pytest.param(["non-existing-word"], [], id="non-existing word")
    ]
)
def test_query_inverted_index_intersect_results(tiny_dataset_fio, query, etalon_answer):
    documents = load_documents(tiny_dataset_fio)
    tiny_inverted_index = build_inverted_index(documents)
    answer = tiny_inverted_index.query(query)
    assert sorted(answer) == sorted(etalon_answer), (
        f"Expected answer is {etalon_answer}, while you got {answer}"
    )


@pytest.fixture()
def sample_documents():
    documents = load_documents(DEFAULT_DATASET_PATH)
    return documents


@pytest.fixture()
def sample_inverted_index(sample_documents):
    inverted_index = build_inverted_index(sample_documents)
    return inverted_index


def test_can_dump_and_load_tiny_inverted_index(tmpdir, sample_inverted_index):
    index_fio = tmpdir.join("./index.dump")
    sample_inverted_index.dump(index_fio, StructStoragePolicy)
    loaded_inverted_index = InvertedIndex.load(index_fio, StructStoragePolicy)
    assert sample_inverted_index == loaded_inverted_index, (
        "load should return the same inverted index"
    )


def test_correctly_process_build(tmpdir, sample_inverted_index):
    index_fio = tmpdir.join("./index.dump")
    process_build(DEFAULT_DATASET_PATH, index_fio)
    loaded_inverted_index = InvertedIndex.load(index_fio, StructStoragePolicy)
    assert sample_inverted_index == loaded_inverted_index, (
        "load should return the same inverted index"
    )


def test_correctly_process_queries_from_file(capsys):
    with open(DEFAULT_QUERY_PATH, 'r') as queries_fin:
        process_queries_from_file(
            inverted_index_filepath=DEFAULT_INVERTED_INDEX_STORE_PATH,
            query_file=queries_fin,
        )
    captured = capsys.readouterr()
    answers = [ans.split(',') for ans in captured.out.split('\n')]
    etalon_answers = [["666", "7"], ["666", "13"], ["666"], [""], [""]]
    for ids, etalon_ids in zip(answers, etalon_answers):
        assert sorted(ids) == sorted(etalon_ids), (
            "answers for queries are not correct"
        )


@pytest.mark.parametrize(
    "queries_list, etalon_answers",
    [
        pytest.param([["fall"]], [["666", "7"]], id="one-list-one-word"),
        pytest.param([["fall", "from"]], [["666"]], id="one-list-two-words"),
        pytest.param([["thing"], ["faith"]], [["123"], ["7"]], id="two-lists-one-word-each"),
        pytest.param([["non-existing-word"]], [[""]], id="non-existing word")
    ]
)
def test_correctly_process_queries_from_cli(capsys, queries_list, etalon_answers):
    process_queries_from_cli(
        inverted_index_filepath=DEFAULT_INVERTED_INDEX_STORE_PATH,
        queries_list=queries_list,
    )
    captured = capsys.readouterr()
    answers = [ans.split(',') for ans in captured.out.split('\n')]
    for ids, etalon_ids in zip(answers, etalon_answers):
        assert sorted(ids) == sorted(etalon_ids), (
            "answers for queries are not correct"
        )


def test_handle_callback_query_for_file_query(capsys):
    with open(DEFAULT_QUERY_PATH, 'r') as queries_fin:
        arguments = Namespace(
            inverted_index_filepath=DEFAULT_INVERTED_INDEX_STORE_PATH,
            query_file=queries_fin,
            query_list=None,
        )
        callback_query(arguments)
    captured = capsys.readouterr()
    answers = [ans.split(',') for ans in captured.out.split('\n')]
    etalon_answers = [["666", "7"], ["666", "13"], ["666"], [""], [""]]
    for ids, etalon_ids in zip(answers, etalon_answers):
        assert sorted(ids) == sorted(etalon_ids), (
            "answers for queries are not correct"
        )


@pytest.mark.parametrize(
    "queries_list, etalon_answers",
    [
        pytest.param([["fall"]], [["666", "7"]], id="one-list-one-word"),
        pytest.param([["fall", "from"]], [["666"]], id="one-list-two-words"),
        pytest.param([["thing"], ["faith"]], [["123"], ["7"]], id="two-lists-one-word-each"),
        pytest.param([["non-existing-word"]], [[""]], id="non-existing word")
    ]
)
def test_handle_callback_query_for_list_query(capsys, queries_list, etalon_answers):
    with open(DEFAULT_QUERY_PATH, 'r') as queries_fin:
        arguments = Namespace(
            inverted_index_filepath=DEFAULT_INVERTED_INDEX_STORE_PATH,
            query_file=queries_fin,
            query_list=queries_list,
        )
        callback_query(arguments)
    captured = capsys.readouterr()
    answers = [ans.split(',') for ans in captured.out.split('\n')]
    for ids, etalon_ids in zip(answers, etalon_answers):
        assert sorted(ids) == sorted(etalon_ids), (
            "answers for queries are not correct"
        )
