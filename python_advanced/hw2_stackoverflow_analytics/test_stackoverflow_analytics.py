from textwrap import dedent
import pytest

from task_Bulanbaev_Artur_stackoverflow_analytics import *


def test_linear_dictionary_correctness():
    ld1, ld2 = LinearDict(), LinearDict()
    ld1['a'] = 2
    ld1['b'] = 3
    ld2['a'] = 4
    ld2['c'] = 5
    true_ld_sum = LinearDict()
    true_ld_sum['a'] = 6
    true_ld_sum['b'] = 3
    true_ld_sum['c'] = 5
    assert true_ld_sum == ld1 + ld2, (
        "Linear dictionaries summation is incorrect"
    )


def test_load_stopwords():
    etalon_stopwords = ['a', 'is', 'than']
    stopwords = load_stopwords(DEFAULT_STOP_WORDS_PATH)
    assert etalon_stopwords == stopwords, (
        f"Expected answer is {etalon_stopwords}, while you got {stopwords}"
    )


def test_load_sample_xml():
    etalon_titles_by_year = {2019: 2, 2020: 1}
    yearly_title_score = YearlyAnalytics.load_from_xml(DEFAULT_POSTS_SAMPLE_PATH)
    titles_by_year = {}
    for year, title_list in yearly_title_score.items():
        titles_by_year[year] = len(title_list)
    assert etalon_titles_by_year == titles_by_year, (
        f"Expected answer is {etalon_titles_by_year}, while you got {titles_by_year}"
    )


def test_make_empty_mapping(tmpdir):
    tiny_xml = dedent("""\
        <row Id="123" PostTypeId="1" CreationDate="2019-10-15T00:44:56.847" Score="10" />
        <row Id="123" PostTypeId="2" CreationDate="2019-10-15" Score="10" Title="Is SEO better?" />
    """)
    dataset_fio = tmpdir.join("dataset.xml")
    dataset_fio.write(tiny_xml)
    yearly_title_score = YearlyAnalytics.load_from_xml(dataset_fio)
    titles_by_year = {}
    for year, title_list in yearly_title_score.items():
        titles_by_year[year] = len(title_list)
    etalon_titles_by_year = {}
    assert etalon_titles_by_year == titles_by_year, (
        f"Expected answer is {etalon_titles_by_year}, while you got {titles_by_year}"
    )


def test_gather_analytics(caplog):
    with caplog.at_level("DEBUG"):
        yearly_title_score = YearlyAnalytics.load_from_xml(DEFAULT_POSTS_SAMPLE_PATH)
        stopwords = load_stopwords(DEFAULT_STOP_WORDS_PATH)
        analytics = YearlyAnalytics.build(yearly_title_score, stopwords)
        assert isinstance(analytics.table, defaultdict), (
            "Returned instance is not a default dictionary"
        )
        assert any("process XML dataset" in message for message in caplog.messages)


def test_load_queries():
    etalon_queries = [[2019, 2019, 2], [2019, 2020, 4]]
    queries = load_queries(DEFAULT_QUERIES_PATH)
    assert sorted(queries) == sorted(etalon_queries), (
        f"Expected answer is {etalon_queries}, while you got {queries}"
    )


@pytest.fixture()
def default_table():
    yearly_title_score = YearlyAnalytics.load_from_xml(DEFAULT_POSTS_SAMPLE_PATH)
    stopwords = load_stopwords(DEFAULT_STOP_WORDS_PATH)
    analytics = YearlyAnalytics.build(yearly_title_score, stopwords)
    return analytics


@pytest.mark.parametrize(
    "query, etalon_answer",
    [
        pytest.param([[2019, 2019, 2]],
                     [{"start": 2019, "end": 2019, "top": [["seo", 15], ["better", 10]]}],
                     id="one year"),
        pytest.param([[2019, 2020, 4]],
                     [{"start": 2019, "end": 2020, "top": [["better", 30], ["javascript", 20],
                                                           ["python", 20], ["seo", 15]]}],
                     id="two years"),
        pytest.param([[2019, 2019, 1], [2019, 2020, 1]],
                     [{"start": 2019, "end": 2019, "top": [["seo", 15]]},
                      {"start": 2019, "end": 2020, "top": [["better", 30]]}],
                     id="many queries"),
        pytest.param([],
                     [],
                     id="no queries"),
    ]
)
def test_respond_to_queries(default_table, query, etalon_answer):
    answer = default_table.respond_to_queries(query)
    etalon_answer = [json.dumps(ans, indent=None) for ans in etalon_answer]
    assert sorted(answer) == sorted(etalon_answer), (
        f"Expected answer is {etalon_answer}, while you got {answer}"
    )


def test_respond_to_queries_from_file(default_table, caplog):
    with caplog.at_level("DEBUG"):
        etalon_answer = [{"start": 2019, "end": 2019, "top": [["seo", 15], ["better", 10]]},
                         {"start": 2019, "end": 2020, "top": [["better", 30], ["javascript", 20],
                                                              ["python", 20], ["seo", 15]]}]
        etalon_answer = [json.dumps(ans, indent=None) for ans in etalon_answer]
        queries = load_queries(DEFAULT_QUERIES_PATH)
        answer = default_table.respond_to_queries(queries)
        assert sorted(answer) == sorted(etalon_answer), (
            f"Expected answer is {etalon_answer}, while you got {answer}"
        )
        assert any("got query" in message for message in caplog.messages)
        assert any("finish processing queries" in message for message in caplog.messages)
        assert all(record.levelno <= logging.WARNING for record in caplog.records)


def test_respond_to_many_queries(default_table, tmpdir, caplog):
    with caplog.at_level("DEBUG"):
        many_queries = dedent("""\
                2019,2019,20
            """)
        many_queries_file = tmpdir.join("many_queries.csv")
        many_queries_file.write(many_queries)
        queries = load_queries(many_queries_file)
        answer = default_table.respond_to_queries(queries)
        assert any("not enough data to answer" in message for message in caplog.messages)
