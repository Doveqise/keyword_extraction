import logging
import multiprocessing
import os.path
import sys
 
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    if len(sys.argv) < 4:
        print(globals()['__doc__'], locals())
        sys.exit(1)
    input_dir, outp1, outp2 = sys.argv[1:4]
    path = PathLineSentences(input_dir)
    model = Word2Vec(path,
                     vector_size=256, window=10, min_count=5,
                     workers=multiprocessing.cpu_count())
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)