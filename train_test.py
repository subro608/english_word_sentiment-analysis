from data_preprocess import preprocess_data, letters
from model import RNN2
import torch.optim as optim
import torch
import torch.nn as nn
import random

all_letters, n_letters = letters()

categories = preprocess_data()
lines = list(categories.keys())
random.shuffle(lines)


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    l = [0, 0, 0]
    l[category_i] = l[category_i] + 1
    return l


def letterToIndex(letter):
    try:
        for i, k in enumerate(all_letters):
            if letter == k:
                return i
    except:
        print('please enter a valid character')


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1

    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    l = [0, 0, 0]
    l[category_i] = l[category_i] + 1
    return l


def train(category_tensors, line_tensors):
    final_loss = 0
    correct_count = 0
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adadelta(params=rnn2.parameters(), lr=0.5)
    for i in range(0, 100):
        line_tensor = line_tensors[i]
        category = category_tensors[i]
        category_tensor = torch.tensor(category, dtype=torch.float)
        hidden = None

        for j in range(line_tensor.size()[0]):
            output, hidden = rnn2(line_tensor[j].view(1, -1).unsqueeze(1), hidden)
        guess_i = categoryFromOutput(output)

        if guess_i == category:
            correct = '✓'
            correct_count = correct_count + 1
        else:
            correct = '✗ (%s)' % category
        loss = criterion(output.view(1, -1), category_tensor.view(1, 3))
        final_loss = final_loss + loss

    final_loss = final_loss / 100

    final_loss.backward()
    optimizer.step()

    return final_loss, correct_count


def evaluate2(line_tensor):
    hidden = None
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn2(line_tensor[i].view(1, -1).unsqueeze(1), hidden)

    return output


def predict(input_line, n_predictions=3):
    output = evaluate2(lineToTensor(input_line))
    guess = categoryFromOutput(output)
    if guess == [1,0,0] :
        print("the word is positive")
    elif guess == [0,1,0] :
        print("the word is negative")
    elif guess == [0,0,1] :
        print("the word is neutral")


if __name__ == "__main__":
    all_letters, n_letters = letters()
    input_size = n_letters
    hidden_size = 50
    output_size = 3
    batch_size = 1
    n_layers = 2
    seq_len = 15
    rnn2 = RNN2(input_size, hidden_size, output_size, n_layers=n_layers)
    n_iters = 100000
    cat_tensors = []
    line_tensors = []
    all_losses = []
    epochs = 3
    optimizer = optim.Adadelta(params=rnn2.parameters(), lr=0.5)

    answer = input("Do you want to test or train?")
    if answer == 'test':
        num_corr2 = 0
        iters = 100000
        for i in range(iters + 1, 110002):

            category = categories[lines[i]]
            print('category {}'.format(category))
            category_tensor = torch.tensor(category, dtype=torch.float)
            line_tensor = lineToTensor(lines[i])
            output = evaluate2(line_tensor)
            guess_new = categoryFromOutput(output)
            if guess_new == category:
                correct = '✓'
                num_corr2 = num_corr2 + 1
            else:
                correct = '✗ (%s)' % category
            print('%d %d%% %s' % (i, i / n_iters * 100, correct))
        print(num_corr2 / 100)
        answer3 = input("Enter your word for sentiment analysis")

        print(predict(answer3, 3))
    else:
        for epoch in range(epochs):
            batch_num = 0
            correct_sum = 0
            final_loss = 0
            rnn2.zero_grad()
            optimizer.zero_grad()

            for iter in range(0, n_iters + 1):

                while True:

                    category = categories[lines[iter]]
                    cat_tensors.append(category)
                    line_tensor = lineToTensor(lines[iter])
                    line_tensors.append(line_tensor)
                    if iter % 100 != 0 or iter == 0:
                        break
                    else:
                        batch_num = batch_num + 1
                        final_loss, correct = train(cat_tensors, line_tensors)
                        correct_sum = correct_sum + correct
                        cat_tensors.clear()
                        line_tensors.clear()
                        if iter != 0:
                            print('%d %s %.4f  %s %s' % (batch_num, iter, final_loss, correct_sum, epoch))
                            break
            print('accuracy is %.6f%%' % (correct_sum / 100000 * 100))
