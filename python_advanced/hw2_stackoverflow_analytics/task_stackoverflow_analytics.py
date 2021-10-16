"""
Provides interface to extract analytical information from stackoverflow questions.
"""

import re
import json
import logging
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from lxml import etree


APPLICATION_NAME = "stackoverflow_analytics"
DEFAULT_POSTS_SAMPLE_PATH = "./tiny_posts_sample.xml"
DEFAULT_STOP_WORDS_PATH = "./tiny_stop_words.txt"
DEFAULT_QUERIES_PATH = "./tiny_queries.csv"
LOG_COMMON = "stackoverflow_analytics.log"
LOG_WARNING = "stackoverflow_analytics.warn"

logger = logging.getLogger(APPLICATION_NAME)


class LinearDict(defaultdict):
    """Performs linear operation over dictionaries (summation by equal keys)"""
    def __init__(self):
        super().__init__(int)

    def __add__(self, other):
        new_dict = LinearDict()
        for key in self.keys():
            new_dict[key] = new_dict[key] + self[key]
        for key in other.keys():
            new_dict[key] = new_dict[key] + other[key]
        return new_dict


class YearlyAnalytics:
    """Provides access to table with aggregated information"""
    def __init__(self, table: defaultdict):
        self.table = table

    def respond_to_queries(self, queries: list) -> list:
        """Return top-n words in period from start_year to end_year in json format"""
        result = []
        for query in queries:
            start_year, end_year, top_n = query
            logger.debug('got query "%d,%d,%d"', start_year, end_year, top_n)
            period_table = LinearDict()
            for year in range(start_year, end_year + 1):
                period_table = period_table + self.table[year]
            period_table = period_table.items()
            if len(period_table) < top_n:
                logger.warning('not enough data to answer, found %d words out of %d for period "%d,%d"',
                               len(period_table), top_n, start_year, end_year)
            period_table = sorted(period_table, key=lambda kv: kv[0], reverse=False)
            period_table = sorted(period_table, key=lambda kv: kv[1], reverse=True)[:top_n]
            response = [[pair[0], pair[1]] for pair in period_table]
            response = {"start": start_year, "end": end_year, "top": response}
            response = json.dumps(response, indent=None)
            result.append(response)
        logger.info("finish processing queries")
        return result

    @classmethod
    def build(cls, yearly_title_score: defaultdict, stopwords: list):
        """Build analytical table from yearly title-score collection"""
        yearly_word_scores = defaultdict(LinearDict)
        for year, title_scores in yearly_title_score.items():
            year_stat = LinearDict()
            for title, score in title_scores:
                title_stat = LinearDict()
                words = re.findall(r"\w+", title.lower())
                words = list(set(words))
                for word in words:
                    if word not in stopwords:
                        title_stat[word] = score
                year_stat = year_stat + title_stat
            yearly_word_scores[year] = year_stat
        logger.info("process XML dataset, ready to serve queries")
        return YearlyAnalytics(table=yearly_word_scores)

    @staticmethod
    def load_from_xml(filepath: str) -> defaultdict:
        """Load questions from xml file and aggregates them by year"""
        yearly_title_score = defaultdict(list)
        with open(filepath, 'r') as file:
            for line in file:
                tree = etree.fromstring(line)
                post_type_id = tree.get('PostTypeId')
                year = int(tree.get('CreationDate')[:4])
                score = int(tree.get('Score'))
                title = tree.get('Title')
                if title and post_type_id == '1':
                    yearly_title_score[year].append([title, score])
        return yearly_title_score


def load_stopwords(filepath: str) -> list:
    """Load stopwords (in koi8-r encoding)"""
    stopwords = []
    with open(filepath, 'r', encoding='koi8-r') as file:
        for line in file:
            stopwords.append(line.strip())
    return stopwords


def load_queries(filepath: str) -> list:
    """Load queries from csv file in form [start_year, end_year, top_n_words]"""
    queries = []
    with open(filepath, 'r') as file:
        for line in file:
            queries.append(list(map(int, line.strip().split(','))))
    return queries


def callback(arguments):
    """Callback function to process CLI commands"""
    yearly_title_score = YearlyAnalytics.load_from_xml(arguments.questions_filepath)
    stopwords = load_stopwords(arguments.stopwords_file)
    analytics = YearlyAnalytics.build(yearly_title_score, stopwords)
    queries = load_queries(arguments.queries_filepath)
    result = analytics.respond_to_queries(queries)
    print('\n'.join(result))


def setup_parser(parser):
    """Parse commands from CLI"""

    parser.add_argument(
        "--questions", dest="questions_filepath",
        default=DEFAULT_POSTS_SAMPLE_PATH,
        help="path to load questions which are used in further analytics",
    )
    parser.add_argument(
        "--stop-words", dest="stopwords_file",
        # type=EncodedFileType("r", encoding="koi8-r"),
        default=DEFAULT_STOP_WORDS_PATH,
        help="stopwords in koi8-r format to filter when responding to queries",
    )
    parser.add_argument(
        "--queries", dest="queries_filepath",
        default=DEFAULT_QUERIES_PATH,
        help="queries to run against yearly analytics table",
    )
    parser.set_defaults(callback=callback)


def setup_logging():
    """"Logging"""
    formatter = logging.Formatter(
        fmt="%(levelname)s: %(message)s",
    )

    file_handler = logging.FileHandler(
        filename=LOG_COMMON,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    file_warning_handler = logging.FileHandler(
        filename=LOG_WARNING,
    )
    file_warning_handler.setLevel(logging.WARNING)
    file_warning_handler.setFormatter(formatter)

    logger = logging.getLogger(APPLICATION_NAME)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(file_warning_handler)


def main():
    """main function"""
    setup_logging()
    parser = ArgumentParser(
        prog="stackoverflow-analytics",
        description="provides analytics for questions on stackoverflow platform",
        # formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
