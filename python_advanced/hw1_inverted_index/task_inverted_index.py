"""
Provides interface to build, load, dump and pass queries to inverted index.
"""

import sys
import struct
from io import TextIOWrapper
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError, FileType

DEFAULT_DATASET_PATH = './tiny_dataset.txt'
DEFAULT_INVERTED_INDEX_STORE_PATH = './inverted.index'
DEFAULT_QUERY_PATH = './queries.txt'


class EncodedFileType(FileType):
    """Improved FileType to handle different encodings"""
    def __call__(self, string):
        # the special argument "-" means sys.std{in,out}
        if string == '-':
            if 'r' in self._mode:
                stdin = TextIOWrapper(sys.stdin.buffer, encoding=self._encoding)
                return stdin
            elif 'w' in self._mode:
                stdout = TextIOWrapper(sys.stdout.buffer, encoding=self._encoding)
                return stdout
            else:
                msg = 'argument "-" with mode %r' % self._mode
                raise ValueError(msg)
        try:
            return open(string, self._mode, self._bufsize, self._encoding, self._errors)
        except OSError as error:
            message = "can't open '%s': %s"
            raise ArgumentTypeError(message % (string, error))


class StoragePolicy:
    """Abstract class for storage policy"""
    @staticmethod
    def dump(word_to_docs_mapping, filepath: str):
        """Abstract method for dumping, overload this in your storage policy"""

    @staticmethod
    def load(filepath: str) -> defaultdict:
        """Abstract method for loading, overload this in your storage policy"""


class StructStoragePolicy(StoragePolicy):
    """Provides storage policy using 'struct' library"""
    @staticmethod
    def dump(word_to_docs_mapping: defaultdict, filepath: str):
        """Dump inverted index into binary file"""
        file_out = open(filepath, 'wb')
        file_out.write(struct.pack('>i', len(word_to_docs_mapping)))
        for term, document_ids in word_to_docs_mapping.items():
            term_size = len(term.encode())
            file_out.write(struct.pack('>h', term_size))
            file_out.write(struct.pack(f'>{term_size}s', term.encode()))
            num_of_ids = len(document_ids)
            file_out.write(struct.pack('>h', num_of_ids))
            for doc_id in document_ids:
                doc_id = int(doc_id)
                file_out.write(struct.pack('>h', doc_id))
        file_out.close()

    @staticmethod
    def load(filepath: str) -> defaultdict:
        """Load inverted index from binary file"""
        file_in = open(filepath, 'rb')
        mapping_size = struct.unpack('>i', file_in.read(struct.calcsize('i')))[0]
        word_to_docs_mapping = defaultdict(list)
        for _ in range(mapping_size):
            term_size = struct.unpack('>h', file_in.read(struct.calcsize('h')))[0]
            term = struct.unpack(f'>{term_size}s', file_in.read(term_size))[0]
            term = term.decode()
            num_of_ids = struct.unpack('>h', file_in.read(struct.calcsize('h')))[0]
            document_ids = []
            for _ in range(num_of_ids):
                doc_id = struct.unpack('>h', file_in.read(struct.calcsize('h')))[0]
                doc_id = str(doc_id)
                document_ids.append(doc_id)
            word_to_docs_mapping[term] = document_ids
        file_in.close()
        return word_to_docs_mapping


class InvertedIndex:
    """Table of terms and documents where the term is present"""
    def __init__(self, index: defaultdict):
        self.index = index

    def query(self, words: list) -> list:
        """Return the list of relevant documents for the given query"""
        assert isinstance(words, list), "query should be provided as list of words"
        document_ids = set(self.index[words[0]])
        for term in words[1:]:
            document_ids &= set(self.index[term])
        return list(document_ids)

    def dump(self, filepath: str, storage_policy: StoragePolicy):
        """Writes inverted index to file using storage policy"""
        storage_policy.dump(self.index, filepath)

    @classmethod
    def load(cls, filepath: str, storage_policy: StoragePolicy):
        """Reads inverted index from file using storage policy"""
        return cls(index=storage_policy.load(filepath))

    def __eq__(self, other):
        return self.index == other.index


def load_documents(filepath: str) -> dict:
    """Load documents from file and save to dict"""
    documents = {}
    with open(filepath, 'r') as file:
        for text in file:
            article_id, article_text = text.strip().split('\t', 1)
            documents[article_id] = article_text
    return documents


def build_inverted_index(documents: dict):
    """Make inverted index from dictionary"""
    dictionary = defaultdict(list)
    for article_id, article_text in documents.items():
        words = article_text.split()
        for term in set(words):
            dictionary[term].append(article_id)
    return InvertedIndex(dictionary)


def callback_build(arguments):
    """Handle build command"""
    return process_build(arguments.dataset, arguments.output)


def process_build(dataset_filepath, inverted_index_filepath):
    """Build inverted index from file and save it"""
    inverted_index = build_inverted_index(load_documents(dataset_filepath))
    inverted_index.dump(inverted_index_filepath, StructStoragePolicy)


def callback_query(arguments):
    """Handle query command"""
    if arguments.query_list:
        process_queries_from_cli(arguments.inverted_index_filepath, arguments.query_list)
    else:
        process_queries_from_file(arguments.inverted_index_filepath, arguments.query_file)


def process_queries_from_file(inverted_index_filepath: str, query_file):
    """Process queries from file"""
    inverted_index = InvertedIndex.load(inverted_index_filepath, StructStoragePolicy)
    for query_line in query_file:
        queries = query_line.strip().split()
        document_ids = inverted_index.query(queries)
        print(','.join(document_ids))


def process_queries_from_cli(inverted_index_filepath: str, queries_list: list):
    """Process queries from CLI"""
    inverted_index = InvertedIndex.load(inverted_index_filepath, StructStoragePolicy)
    for queries in queries_list:
        document_ids = inverted_index.query(queries)
        print(','.join(document_ids))


def setup_parser(parser):
    """Parse commands from CLI"""
    subparsers = parser.add_subparsers(help="choose command")

    build_parser = subparsers.add_parser(
        "build",
        help="build inverted index and save in binary format to hard drive",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    build_parser.add_argument(
        "-d", "--dataset",
        default=DEFAULT_DATASET_PATH,
        help="path to dataset to load",
    )
    build_parser.add_argument(
        "-o", "--output",
        default=DEFAULT_INVERTED_INDEX_STORE_PATH,
        help="path to store inverted index in a binary format",
    )
    build_parser.set_defaults(callback=callback_build)

    query_parser = subparsers.add_parser(
        "query",
        help="query inverted index",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    query_parser.add_argument(
        "-i", "--index",
        dest='inverted_index_filepath',
        default=DEFAULT_INVERTED_INDEX_STORE_PATH,
        help="path to read inverted index in a binary form",
    )
    query_file_group = query_parser.add_mutually_exclusive_group(required=False)
    query_file_group.add_argument(
        "--query-file-utf8", dest="query_file",
        type=EncodedFileType("r", encoding="utf-8"),
        default=TextIOWrapper(sys.stdin.buffer, encoding="utf-8"),
        help="query in utf8 to run against inverted index",
    )
    query_file_group.add_argument(
        "--query-file-cp1251", dest="query_file",
        type=EncodedFileType("r", encoding="cp1251"),
        default=TextIOWrapper(sys.stdin.buffer, encoding="cp1251"),
        help="query in cp1251 to run against inverted index",
    )
    query_parser.add_argument(
        "-q", "--query",
        dest='query_list', nargs="*", action='append',
        help="query to run against inverted index",
    )
    query_parser.set_defaults(callback=callback_query)


# def process_arguments(dataset: str, query: list):
#     documents = load_documents(dataset)
#     inverted_index = build_inverted_index(documents)
#     query = query.split()
#     document_ids = inverted_index.query(query)
#     return document_ids


def main():
    """main function"""
    parser = ArgumentParser(
        prog="inverted-index",
        description="tool to build, dump, load and query inverted index",
        # formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
