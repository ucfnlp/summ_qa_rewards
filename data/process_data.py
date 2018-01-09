import json
import os
import sys

import hashlib

import data_args


reload(sys)
sys.setdefaultencoding('utf8')


def process_data(args):
    train, dev, test, unique_w = split_data(args)

    if args.pipeline: # takes a long-o-time
        core_nlp(args, train[0], dev[0], test[0])
    else:
        word_counts = [150000, 100000, 50000]

        for count in word_counts:
            print 'Building dataset for vocab size : ' + str(count)
            vocab = create_vocab_map(unique_w, count)
            machine_ready(args, train, dev, test, vocab, count)


def split_data(args):
    small_size_counter = 0

    unique_words = dict() # word : count

    highlights_train = []
    articles_train = []

    highlights_dev = []
    articles_dev = []

    highlights_test = []
    articles_test = []

    train_urls, dev_urls, test_urls = get_url_sets(args)

    data_dirs = [args.raw_data_cnn, args.raw_data_dm]

    for raw_data in data_dirs:
        for subdir, dirs, files in os.walk(raw_data):
            for file_in in files:

                current_article = []
                current_highlights = []

                if file_in.startswith('.'):
                    continue

                sha = file_in.split('.')[0]
                file_in = open(subdir + '/' + file_in, 'r')
                incoming_hl = False

                for line in file_in:
                    if len(line.strip()) == 0:
                        continue

                    if '@highlight' in line:
                        incoming_hl = True
                        continue

                    if incoming_hl:
                        current_highlights.append(line)
                        incoming_hl = False
                    else:
                        current_article.append(line)

                current_article, current_highlights = tokenize(args, current_article, current_highlights, unique_words)

                if len(current_article) == 0:
                    continue

                catg = get_set(sha, train_urls, dev_urls, test_urls)

                if catg < 0:
                    print 'Problem with : ' + str(sha)
                    continue

                if catg == 1: #TRAIN
                    highlights_train.append(current_highlights)
                    articles_train.append(current_article)
                elif catg == 2: #DEV
                    highlights_dev.append(current_highlights)
                    articles_dev.append(current_article)
                else:#TEST
                    highlights_test.append(current_highlights)
                    articles_test.append(current_article)

                small_size_counter += 1

                if not args.full_test and small_size_counter >= args.small_limit:
                    return (highlights_train, articles_train), (highlights_dev, articles_dev), (
                    highlights_test, articles_test), unique_words

    return (highlights_train, articles_train), (highlights_dev, articles_dev), (highlights_test, articles_test), unique_words


def core_nlp(args, train, dev, test):

    ofp_train = open(args.intermediate + '_train.txt', 'w+')
    ofp_dev = open(args.intermediate + '_dev.txt', 'w+')
    ofp_test = open(args.intermediate + '_test.txt', 'w+')

    for highlight in train:

        for sentence in highlight:
            for word in sentence:
                ofp_train.write(word + ' ')
            ofp_train.write('.\n')
        ofp_train.write('\n')

    ofp_train.close()

    for highlight in dev:

        for sentence in highlight:
            for word in sentence:
                ofp_dev.write(word + ' ')
            ofp_dev.write('.\n')
        ofp_dev.write('\n')

    ofp_dev.close()

    for highlight in test:

        for sentence in highlight:
            for word in sentence:
                ofp_test.write(word + ' ')
            ofp_test.write('.\n')
        ofp_test.write('\n')

    ofp_test.close()


def seqs_art(args, inp, vocab, entity_set, raw_entity_mapping, first_word_map):
    inp_seqs = []
    inp_ents = []

    for article in inp:
        single_inp_seqs = []
        entities_in_article = set()

        for sent in article:

            single_inp_sent = []
            for w in xrange(len(sent)):

                # 1.) check if word in vocab or not
                word = sent[w].lower()

                index = vocab[word] if word in vocab else 1
                single_inp_sent.append(index)

                # 2.) check if word starts NER
                if word in first_word_map:

                    originals = first_word_map[word]

                    for raw_text_entity in originals:

                        text_ls = raw_text_entity.split(' ')
                        ent_len = len(text_ls)
                        entity_found = True

                        for i in xrange(ent_len):
                            if w + i >= len(sent):
                                entity_found = False
                                break

                            if sent[w + i] != text_ls[i]:
                                entity_found = False
                                break

                        if entity_found:
                            entity = entity_set[raw_entity_mapping[raw_text_entity]]
                            if entity[0] not in entities_in_article and entity[1] != 'ROOT':
                                entities_in_article.add(entity[0])
                            break

            single_inp_seqs.append(single_inp_sent)
        inp_ents.append(list(entities_in_article))
        inp_seqs.append(single_inp_seqs)

    return inp_seqs, inp_ents


def seqs_hl(args, inp, vocab, entity_set, entity_counter, raw_entity_mapping, first_word_map,  type):
    input_hl_seqs = []
    input_hl_entities = []

    tag_ls = ['PERSON', 'LOCATION', 'ORGANIZATION', 'MISC']

    annotated_hl_fp = open(args.intermediate + '_' + str(type) + '.txt.json', 'r')
    annotated_hl_json = json.load(annotated_hl_fp)
    sentences = annotated_hl_json['sentences']

    hl_idx_start = hl_idx_end = 0

    total_samples = len(inp)
    print total_samples, 'total samples'

    for sample in xrange(total_samples):
        if (total_samples / 10) > 0 and sample % (total_samples / 10) == 0:
            print '..', sample

        single_inp_hl = []
        single_inp_hl_entity_map = []

        highlight = inp[sample]

        hl_idx_start = hl_idx_end
        hl_idx_end += len(highlight)

        for h in range(hl_idx_start, hl_idx_end):

            single_sent_hl_entity_map = []

            # 1.) find sentence root
            working_anno_hl = sentences[h]
            basic_dep = working_anno_hl['basicDependencies']
            tokens_ls = working_anno_hl['tokens']

            root_basic_dep = basic_dep[0]
            root_idx = root_basic_dep['dependent']

            root_token= tokens_ls[root_idx - 1]
            root_lemma = root_token['lemma']

            if root_lemma.lower() not in entity_set: # previously not found @entity
                entity_info = [entity_counter, root_token['ner']]
                entity_set[root_lemma.lower()] = entity_info
                entity_counter += 1

            clean_hl_vec = create_hl_vector(args, vocab, tokens_ls)

            hl_vec = clean_hl_vec[:]
            hl_vec[root_idx - 1] = 2

            single_inp_hl.append(hl_vec)
            single_sent_hl_entity_map.append(entity_set[root_lemma.lower()])

            # 2.) find all instances of tags
            # named entities in the form : (entity name, start, end, type, raw name, first word)
            entities = find_ner_tokens(tokens_ls, tag_ls)

            for ner in entities:
                if ner[0] not in entity_set:
                    entity_info = [entity_counter, ner[3]]
                    entity_set[ner[0]] = entity_info
                    entity_counter += 1

                hl_vec_complete = clean_hl_vec[:ner[1]] + [args.placeholder] + clean_hl_vec[ner[2] + 1:]

                single_inp_hl.append(hl_vec_complete)
                single_sent_hl_entity_map.append(entity_set[ner[0]])

                if ner[4] not in raw_entity_mapping:
                    raw_entity_mapping[ner[4]] = ner[0]

                if ner[5] not in first_word_map:
                    first_word_map[ner[5]] = [ner[4]]
                else:
                    originals = first_word_map[ner[5]]

                    if ner[4] not in originals:
                        first_word_map[ner[5]].append(ner[4])

            input_hl_seqs.append(single_inp_hl)
            single_inp_hl_entity_map.append(single_sent_hl_entity_map)

        input_hl_entities.append(single_inp_hl_entity_map)

    return input_hl_seqs, input_hl_entities, entity_counter


def machine_ready(args, train, dev, test, vocab, count):
    entity_set = dict()
    raw_entity_mapping = dict()
    first_word_map = dict()

    entity_counter = 1

    print 'Train data NER HL proc..'
    seqs_train_hl, seqs_train_e, entity_counter = seqs_hl(args, train[0], vocab, entity_set, entity_counter, raw_entity_mapping,
                                          first_word_map, 'train')
    print 'Dev data NER HL proc..'
    seqs_dev_hl, seqs_dev_e, entity_counter = seqs_hl(args, dev[0], vocab, entity_set, entity_counter, raw_entity_mapping,
                                      first_word_map, 'dev')
    print 'Test data NER HL proc..'
    seqs_test_hl, seqs_test_e, entity_counter = seqs_hl(args, test[0], vocab, entity_set, entity_counter, raw_entity_mapping,
                                        first_word_map, 'test')

    print 'Train data indexing..'
    seqs_train_articles, seq_train_art_ents = seqs_art(args, train[1], vocab, entity_set, raw_entity_mapping,
                                                       first_word_map)
    print 'Dev data indexing..'
    seqs_dev_articles, seq_dev_art_ents = seqs_art(args, dev[1], vocab, entity_set, raw_entity_mapping,
                                                   first_word_map)
    print 'Test data indexing..'
    seqs_test_articles, seq_test_art_ents = seqs_art(args, test[1], vocab, entity_set, raw_entity_mapping,
                                                     first_word_map)

    filename_train = args.train if args.full_test else "small_" + args.train
    filename_train = str(count) + '_' + filename_train

    ofp_train = open(filename_train, 'w+')
    final_json_train = dict()

    final_json_train['x'] = seqs_train_articles
    final_json_train['y'] = seqs_train_hl
    final_json_train['e'] = seqs_train_e
    final_json_train['valid_e'] = seq_train_art_ents

    json.dump(final_json_train, ofp_train)
    ofp_train.close()

    filename_dev = args.dev if args.full_test else "small_" + args.dev
    filename_dev = str(count) + '_' + filename_dev

    ofp_dev = open(filename_dev, 'w+')
    final_json_dev = dict()

    final_json_dev['x'] = seqs_dev_articles
    final_json_dev['y'] = seqs_dev_hl
    final_json_dev['e'] = seqs_dev_e
    final_json_dev['valid_e'] = seq_dev_art_ents

    json.dump(final_json_dev, ofp_dev)
    ofp_dev.close()

    filename_test = args.dev if args.full_test else "small_" + args.test
    filename_test = str(count) + '_' + filename_test

    ofp_test = open(filename_test, 'w+')
    final_json_test = dict()

    final_json_test['x'] = seqs_test_articles
    final_json_test['y'] = seqs_test_hl

    json.dump(final_json_test, ofp_test)
    ofp_test.close()

    filename_entities = 'entities.json' if args.full_test else "small_entities.json"
    filename_entities = str(count) + '_' + filename_entities

    ofp_entities = open(filename_entities, 'w+')
    final_json_entities = dict()
    final_json_entities['entities'] = entity_set.items()

    json.dump(final_json_entities, ofp_entities)
    ofp_entities.close()


def tokenize(args, current_article, current_highlights, unique_w):
    article = []
    highlights = []

    for item in current_article:
        sentence = str(item.encode('utf-8'))

        words = sentence.rstrip().split(' ')
        s = []
        for w in words:
            w = w.lower()

            if w in unique_w:
                unique_w[w] += 1
            else:
                unique_w[w] = 1
            s.append(w)

        article.append(s)

    for item in current_highlights:
        sentence = str(item.encode('utf-8'))

        words = sentence.rstrip().split(' ')
        s = []
        for w in words:
            s.append(w)

            w = w.lower()
            if w in unique_w:
                unique_w[w] += 1
            else:
                unique_w[w] = 1


        highlights.append(s)

    return article, highlights


def create_vocab_map(unique_w, count):
    ofp = open('vocab_' + str(count) + '.txt', 'w+')
    vocab_map = dict()
    index = 2

    inv_map = dict()

    for k, v in unique_w.iteritems():
        if v in inv_map:
            inv_map[v].append(k)
        else:
            inv_map[v] = [k]

    ofp.write('<padding>\n<unk>\n<placeholder>\n')

    vocab_map['<padding>'] = 0
    vocab_map['<unk>'] = 1
    vocab_map['<placeholder>'] = 2

    for c in sorted(inv_map.iterkeys(), reverse=True):
        words = inv_map[c]

        for w in words:

            ofp.write(w + '\n')
            vocab_map[w] = index
            index += 1

            if index > count:
                ofp.close()
                return vocab_map

    ofp.close()
    return vocab_map


def get_url_sets(args):

    sha1 = hashlib.sha1

    train_urls = set()
    dev_urls = set()
    test_urls = set()

    train_ofp = open(args.train_urls, 'r')
    dev_ofp = open(args.dev_urls, 'r')
    test_ofp = open(args.test_urls, 'r')

    for line in train_ofp:
        train_urls.add(sha1(line.rstrip()).hexdigest())

    for line in dev_ofp:
        dev_urls.add(sha1(line.rstrip()).hexdigest())

    for line in test_ofp:
        test_urls.add(sha1(line.rstrip()).hexdigest())

    train_ofp.close()
    dev_ofp.close()
    test_ofp.close()

    return train_urls, dev_urls, test_urls


def get_set(file_in, train_urls, dev_urls, test_urls):
    if file_in in train_urls:
        return 1
    elif file_in in dev_urls:
        return 2
    elif file_in in test_urls:
        return 3
    else:
        return -1


def create_hl_vector(args, vocab, tokens_ls):
    vector = []

    for token in tokens_ls:

        word = token['word'].lower()

        if word in vocab:
            vector.append(vocab[word])
        else:
            vector.append(args.unk)

    return vector


def find_ner_tokens(tokens_ls, tag_ls):
    ner_set = set()
    current_ner = None
    start_idx = end_idx = -1

    for i in xrange(len(tokens_ls)):

        item = tokens_ls[i]

        if item['ner'] in tag_ls:

            if current_ner is None:
                start_idx = i
                current_ner = item['ner']
            elif current_ner != item['ner']:
                end_idx = i - 1

                name = ''
                name_raw = ''

                for j in range(start_idx, i):
                    name += tokens_ls[j]['lemma'].lower()
                    name += '' if j == i - 1 else ' '

                    name_raw += tokens_ls[j]['originalText'].lower()
                    name_raw += '' if j == i - 1 else ' '

                # (entity name, start, end, type, raw_name, first word)
                fw = tokens_ls[start_idx]['originalText'].lower()
                ner = (name, start_idx, end_idx, current_ner, name_raw, fw)
                ner_set.add(ner)


                start_idx = i
                current_ner = item['ner']

        elif current_ner is not None:
            end_idx = i - 1

            name = ''
            name_raw = ''

            for j in range(start_idx, i):
                name += tokens_ls[j]['lemma'].lower()
                name += '' if j == i - 1 else ' '

                name_raw += tokens_ls[j]['originalText'].lower()
                name_raw += '' if j == i - 1 else ' '

            fw = tokens_ls[start_idx]['originalText'].lower()
            ner = (name, start_idx, end_idx, current_ner, name_raw, fw)
            ner_set.add(ner)

            start_idx = -1
            current_ner = None

    return ner_set


if __name__ == '__main__':
    args = data_args.get_args()
    process_data(args)
