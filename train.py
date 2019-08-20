import random
import torch

def calculate_accuracy(fx, y):
    """
    Calculate top-1 accuracy

    fx = [batch size, output dim]
     y = [batch size]
    """
    pred_idxs = fx.max(1, keepdim=True)[1]
    correct = pred_idxs.eq(y.view_as(pred_idxs)).sum()
    acc = correct.float() / pred_idxs.shape[0]
    return acc


def calculate_f1(fx, y, idx2target):
    """
    Calculate precision, recall and F1 score
    - Takes top-1 predictions
    - Converts to strings
    - Splits into sub-tokens
    - Calculates TP, FP and FN
    - Calculates precision, recall and F1 score

    fx = [batch size, output dim]
     y = [batch size]
    """
    pred_idxs = fx.max(1, keepdim=True)[1]
    pred_names = [idx2target[i.item()] for i in pred_idxs]
    original_names = [idx2target[i.item()] for i in y]
    true_positive, false_positive, false_negative = 0, 0, 0
    for p, o in zip(pred_names, original_names):
        predicted_subtokens = p.split('|')
        original_subtokens = o.split('|')
        for subtok in predicted_subtokens:
            if subtok in original_subtokens:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in original_subtokens:
            if not subtok in predicted_subtokens:
                false_negative += 1
    try:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision, recall, f1 = 0, 0, 0
    return precision, recall, f1


def parse_line(line):
    """
    Takes a string 'x y1,p1,z1 y2,p2,z2 ... yn,pn,zn and splits into name (x) and tree [[y1,p1,z1], ...]
    """
    name, *tree = line.split(' ')
    tree = [t.split(',') for t in tree if t != '' and t != '\n']
    print(name)
    # array = line.strip().split(' ')
    # name = array[0]
    # if len(array) > 1:
    #     tree = array[1:]
    # else:
    #     tree = []
    return name, tree


def file_iterator(file_path, MAX_LENGTH):
    """
    Takes a file path and creates and iterator
    For each line in the file, parse into a name and tree
    Pad tree to maximum length
    Yields example:
    - example_name = 'target'
    - example_body = [['left_node','path','right_node'], ...]
    """
    with open(file_path, 'rb') as file:
        for line in file:
            line = line.decode('utf-8', 'surrogatepass').strip('\n').strip(' ').split(' ')
            line = line[0:(len(line)-1)]
            # each line is an example

            # each example is made of the function name and then a sequence of triplets
            # the triplets are (node, path, node)

            # example_name, example_body = parse_line(line)
            example_name = line[0]
            if len(line) > 1:
                example_body = line[1:]
            else:
                example_body = []

            # max length set while preprocessing, make sure none longer

            example_length = len(example_body)

            # assert example_length <= MAX_LENGTH

            # #need to pad all to maximum length

            # example_body += [['<pad>', '<pad>', '<pad>']]*(MAX_LENGTH - example_length)

            if example_length <= MAX_LENGTH:
                # need to pad all to maximum length
                # example_body += [['<pad>', '<pad>', '<pad>']]*(MAX_LENGTH - example_length)
                example_body += ['<pad>, <pad>, <pad>'] * (MAX_LENGTH - example_length)
            else:
                example_body = example_body[0: MAX_LENGTH]
            assert len(example_body) == MAX_LENGTH

            yield example_name, example_body, example_length


def numericalize(examples, n, BATCH_SIZE, MAX_LENGTH, word2idx, path2idx, target2idx):
    """
    Examples are a list of list of lists, i.e. examples[0] = [['left_node','path','right_node'], ...]
    n is how many batches we are getting out of `examples`

    Get a batch of raw (still strings) examples
    Create tensors to store them all
    Numericalize each raw example within the batch and convert whole batch tensor
    Yield tensor batch
    """

    assert n * BATCH_SIZE <= len(examples)

    for i in range(n):

        # get the raw data

        raw_batch_name, raw_batch_body, batch_lengths = zip(*examples[BATCH_SIZE * i:BATCH_SIZE * (i + 1)])
        # create a tensor to store the batch

        tensor_n = torch.zeros(BATCH_SIZE).long()  # name
        tensor_l = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long()  # left node
        tensor_p = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long()  # path
        tensor_r = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long()  # right node
        mask = torch.ones((BATCH_SIZE, MAX_LENGTH)).float()  # mask

        # for each example in our raw data

        for j, (name, body, length) in enumerate(zip(raw_batch_name, raw_batch_body, batch_lengths)):
            # convert to idxs using vocab
            # use <unk> tokens if item doesn't exist inside vocab
            temp_n = target2idx.get(name, target2idx['<unk>'])
            # temp_l, temp_p, temp_r = zip(*[(word2idx.get(l, word2idx['<unk>']), path2idx.get(p, path2idx['<unk>']), word2idx.get(r, word2idx['<unk>'])) for l, p, r in body])
            temp_l = []
            temp_p = []
            temp_r = []
            for item in body:
                l, p, r = item.split(',')
                temp_l.append(word2idx.get(l, word2idx['<unk>']))
                temp_p.append(path2idx.get(p, path2idx['<unk>']))
                temp_r.append(word2idx.get(r, word2idx['<unk>']))

            # store idxs inside tensors
            tensor_n[j] = temp_n
            tensor_l[j, :] = torch.LongTensor(temp_l)
            tensor_p[j, :] = torch.LongTensor(temp_p)
            tensor_r[j, :] = torch.LongTensor(temp_r)

            # create masks
            mask[j, length:] = 0

        yield tensor_n, tensor_l, tensor_p, tensor_r, mask


def get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, idx2target, optimizer=None):
    """
    Takes inputs, calculates loss, accuracy and other metrics, then calculates gradients and updates parameters

    if optimizer is None, then we are doing evaluation so no gradients are calculated and no parameters are updated
    """

    if optimizer is not None:
        optimizer.zero_grad()

    fx = model(tensor_l, tensor_p, tensor_r)

    loss = criterion(fx, tensor_n)

    acc = calculate_accuracy(fx, tensor_n)
    precision, recall, f1 = calculate_f1(fx, tensor_n, idx2target)

    if optimizer is not None:
        loss.backward()
        optimizer.step()

    return loss.item(), acc.item(), precision, recall, f1


def train(model, file_path, optimizer, criterion, BATCH_SIZE, CHUNKS,
          LOG_EVERY, LOG_PATH, MAX_LENGTH, idx2target, word2idx, path2idx, target2idx, n_training_examples, device):
    """
    Training loop for the model
    Dataset is too large to fit in memory, so we stream it
    Get BATCH_SIZE * CHUNKS examples at a time (default = 1024 * 10 = 10,240)
    Shuffle the BATCH_SIZE * CHUNKS examples
    Convert raw string examples into numericalized tensors
    Get metrics and update model parameters

    Once we near end of file, may have less than BATCH_SIZE * CHUNKS examples left, but still want to use
    So we calculate number of remaining whole batches (len(examples)//BATCH_SIZE) then do that many updates
    """

    n_batches = 0

    epoch_loss = 0
    epoch_acc = 0
    epoch_r = 0
    epoch_p = 0
    epoch_f1 = 0

    model.train()

    examples = []

    for example_name, example_body, example_length in file_iterator(file_path, MAX_LENGTH):

        examples.append((example_name, example_body, example_length))

        if len(examples) >= (BATCH_SIZE * CHUNKS):

            random.shuffle(examples)

            for tensor_n, tensor_l, tensor_p, tensor_r, mask in numericalize(examples, CHUNKS, BATCH_SIZE, MAX_LENGTH, word2idx, path2idx, target2idx):

                # place on gpu

                tensor_n = tensor_n.to(device)
                tensor_l = tensor_l.to(device)
                tensor_p = tensor_p.to(device)
                tensor_r = tensor_r.to(device)

                # put into model
                loss, acc, p, r, f1 = get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, idx2target, optimizer)
                print(f1)
                epoch_loss += loss
                epoch_acc += acc
                epoch_p += p
                epoch_r += r
                epoch_f1 += f1

                n_batches += 1

                if n_batches % LOG_EVERY == 0:
                    loss = epoch_loss / n_batches
                    acc = epoch_acc / n_batches
                    precision = epoch_p / n_batches
                    recall = epoch_r / n_batches
                    f1 = epoch_f1 / n_batches

                    log = f'\t| Batches: {n_batches} | Completion: {((n_batches * BATCH_SIZE) / n_training_examples) * 100:03.3f}% |\n'
                    log += f'\t| Loss: {loss:02.3f} | Acc.: {acc:.3f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}'
                    with open(LOG_PATH, 'a+') as f:
                        f.write(log + '\n')
                    print(log)

            examples = []

        else:
            pass

    # outside of `file_iterator`, but will probably still have some examples left over
    random.shuffle(examples)

    # get amount of batches we have left
    n = len(examples) // BATCH_SIZE

    # train with remaining batches
    for tensor_n, tensor_l, tensor_p, tensor_r, mask in numericalize(examples, n, BATCH_SIZE, MAX_LENGTH, word2idx, path2idx, target2idx):
        # place on gpu

        tensor_n = tensor_n.to(device)
        tensor_l = tensor_l.to(device)
        tensor_p = tensor_p.to(device)
        tensor_r = tensor_r.to(device)

        # put into model

        loss, acc, p, r, f1 = get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, idx2target, optimizer)

        epoch_loss += loss
        epoch_acc += acc
        epoch_p += p
        epoch_r += r
        epoch_f1 += f1

        n_batches += 1

    return epoch_loss / n_batches, epoch_acc / n_batches, epoch_p / n_batches, epoch_r / n_batches, epoch_f1 / n_batches


def evaluate(model, file_path, criterion, BATCH_SIZE, CHUNKS,
          LOG_EVERY, LOG_PATH, MAX_LENGTH, idx2target, word2idx, path2idx, target2idx, device):
    """
    Similar to training loop, but we do not pass optimizer to get_metrics
    Also wrap get_metrics in `torch.no_grad` to avoid calculating gradients
    """

    n_batches = 0

    epoch_loss = 0
    epoch_acc = 0
    epoch_r = 0
    epoch_p = 0
    epoch_f1 = 0

    model.eval()

    examples = []

    for example_name, example_body, example_length in file_iterator(file_path, MAX_LENGTH):

        examples.append((example_name, example_body, example_length))

        if len(examples) >= (BATCH_SIZE * CHUNKS):

            random.shuffle(examples)

            for tensor_n, tensor_l, tensor_p, tensor_r, mask in numericalize(examples, CHUNKS, BATCH_SIZE, MAX_LENGTH, word2idx, path2idx, target2idx):

                # place on gpu

                tensor_n = tensor_n.to(device)
                tensor_l = tensor_l.to(device)
                tensor_p = tensor_p.to(device)
                tensor_r = tensor_r.to(device)

                # put into model
                with torch.no_grad():
                    loss, acc, p, r, f1 = get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, idx2target)

                epoch_loss += loss
                epoch_acc += acc
                epoch_p += p
                epoch_r += r
                epoch_f1 += f1

                n_batches += 1

                if n_batches % LOG_EVERY == 0:
                    loss = epoch_loss / n_batches
                    acc = epoch_acc / n_batches
                    precision = epoch_p / n_batches
                    recall = epoch_r / n_batches
                    f1 = epoch_f1 / n_batches

                    log = f'\t| Batches: {n_batches} |\n'
                    log += f'\t| Loss: {loss:02.3f} | Acc.: {acc:.3f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}'
                    with open(LOG_PATH, 'a+') as f:
                        f.write(log + '\n')
                    print(log)

            examples = []

        else:
            pass

    # outside of for line in f, but will still have some examples left over

    random.shuffle(examples)

    n = len(examples) // BATCH_SIZE

    for tensor_n, tensor_l, tensor_p, tensor_r, mask in numericalize(examples, n, BATCH_SIZE, MAX_LENGTH, word2idx, path2idx, target2idx):
        # place on gpu

        tensor_n = tensor_n.to(device)
        tensor_l = tensor_l.to(device)
        tensor_p = tensor_p.to(device)
        tensor_r = tensor_r.to(device)

        # put into model
        with torch.no_grad():
            loss, acc, p, r, f1 = get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, idx2target)

        epoch_loss += loss
        epoch_acc += acc
        epoch_p += p
        epoch_r += r
        epoch_f1 += f1

        n_batches += 1
    print(n_batches)

    return epoch_loss / n_batches, epoch_acc / n_batches, epoch_p / n_batches, epoch_r / n_batches, epoch_f1 / n_batches


def numericalize_mm(examples, MAX_LENGTH, word2idx, path2idx, target2idx):
    """
    Examples are a list of list of lists, i.e. examples[0] = [['left_node','path','right_node'], ...]
    n is how many batches we are getting out of `examples`

    Get a batch of raw (still strings) examples
    Create tensors to store them all
    Numericalize each raw example within the batch and convert whole batch tensor
    Yield tensor batch
    """

    # get the raw data

    raw_batch_name, raw_batch_body, batch_lengths = zip(*examples)
    # create a tensor to store the batch

    tensor_n = torch.zeros(len(examples)).long()  # name
    tensor_l = torch.zeros((len(examples), MAX_LENGTH)).long()  # left node
    tensor_p = torch.zeros((len(examples), MAX_LENGTH)).long()  # path
    tensor_r = torch.zeros((len(examples), MAX_LENGTH)).long()  # right node
    mask = torch.ones((len(examples), MAX_LENGTH)).float()  # mask

    # for each example in our raw data

    for j, (name, body, length) in enumerate(zip(raw_batch_name, raw_batch_body, batch_lengths)):
        # convert to idxs using vocab
        # use <unk> tokens if item doesn't exist inside vocab
        temp_n = target2idx.get(name, target2idx['<unk>'])
        # temp_l, temp_p, temp_r = zip(*[(word2idx.get(l, word2idx['<unk>']), path2idx.get(p, path2idx['<unk>']), word2idx.get(r, word2idx['<unk>'])) for l, p, r in body])
        temp_l = []
        temp_p = []
        temp_r = []
        for item in body:
            l, p, r = item.split(',')
            temp_l.append(word2idx.get(l, word2idx['<unk>']))
            temp_p.append(path2idx.get(p, path2idx['<unk>']))
            temp_r.append(word2idx.get(r, word2idx['<unk>']))

        # store idxs inside tensors
        tensor_n[j] = temp_n
        tensor_l[j, :] = torch.LongTensor(temp_l)
        tensor_p[j, :] = torch.LongTensor(temp_p)
        tensor_r[j, :] = torch.LongTensor(temp_r)

        # create masks
        mask[j, length:] = 0

    return tensor_n, tensor_l, tensor_p, tensor_r, mask
