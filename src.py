from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from typing import List
import warnings
import numpy as np
from pathlib import Path
import preprocessor.preprocess

from pyyoutube import Api
import pandas as pd
import googleapiclient.discovery
from langdetect import detect


def detect_lang(text):
    try:
        lang = detect(text)
    except:
        lang = ''
    return lang


def clean_text(row, col_name):
    prep = preprocessor.preprocess.Preprocess()
    text = row[col_name]
    text = prep.clean(text.replace("#", "# "), None)
    return text


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def Encode_TextWithAttention(sentence, tokenizer, maxlen, padding_type='max_length', attention_mask_flag=True):
    encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=maxlen, truncation=True,
                                         padding=padding_type, return_attention_mask=attention_mask_flag)
    return encoded_dict['input_ids'], encoded_dict['attention_mask']


def Encode_TextWithoutAttention(sentence, tokenizer, maxlen, padding_type='max_length', attention_mask_flag=False):
    encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=maxlen, truncation=True,
                                         padding=padding_type, return_attention_mask=attention_mask_flag)
    return encoded_dict['input_ids']


def get_TokenizedTextWithAttentionMask(sentenceList, tokenizer, MAX_LEN=100):
    token_ids_list, attention_mask_list = [], []
    for sentence in sentenceList:
        token_ids, attention_mask = Encode_TextWithAttention(sentence, tokenizer, MAX_LEN)
        token_ids_list.append(token_ids)
        attention_mask_list.append(attention_mask)
    return token_ids_list, attention_mask_list


class supervised_bert_classifier:
    def __init__(self,
                 number_of_classes: int,
                 max_length: int,
                 language: str,
                 use_cuda: bool):
        warnings.warn("Initializing the class might need few minutes because the class must once download and " +
                      "save the required BERT models from the Huggingface servers.\n" +
                      "The pre-trained models will be saved in 'cache_dir' folder.")
        self.number_of_classes = number_of_classes
        self.language = language
        self.max_length = max_length
        self.use_cuda = use_cuda

        self.list_of_models = {"en": 'bert-base-uncased',
                               "de": 'bert-base-german-uncased',
                               "nl": 'wietsedv/bert-base-dutch-uncased',
                               "fi": 'TurkuNLP/bert-base-finnish-uncased-v1',
                               "others": 'bert-base-multilingual-uncased'}

        if self.language not in ["en", "de", "nl", "fi"]:
            warnings.warn(f"You have selected the following language: '{self.language}'.\n" +
                          "However, this class is optimized for one of the following languages " +
                          "'en', 'de', 'nl', and 'fi'.\nThe performance of the classifier for the selcted language " +
                          "might be limited.")
            self.language = 'others'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0 and self.device.type == 'cuda' and self.use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f'using {self.device}')
        if self.device == 'cuda':
            self.model = BertForSequenceClassification.from_pretrained(self.list_of_models[self.language],
                                                                       num_labels=self.number_of_classes).cuda()
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.list_of_models[self.language],
                                                                       num_labels=self.number_of_classes)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
        self.tokenizer = BertTokenizer.from_pretrained(self.list_of_models[self.language], do_lower_case=True)

        self.train_dataloader = None
        self.validation_dataloader = None
        self.train_history = self.valid_history = None
        self.trained = False
        self.loaded = False

    def add_data(self,
                 train_sentences: List[str],
                 train_labels: List[int],
                 val_sentences: List[str],
                 val_labels: List[int],
                 batch_size: int = 16):
        if self.train_dataloader is not None or self.validation_dataloader is not None:
            warnings.warn('The class has already some data. The old data will be replaced by the new data.')
        else:
            warnings.warn('Adding data takes few seconds.')
        if not isinstance(train_sentences, list) or not isinstance(train_labels, list) or not isinstance(
                val_sentences, list) or not isinstance(val_labels, list):
            raise ValueError("The sentences and labels must be lists.")
        if not len(train_sentences) == len(train_labels):
            raise ValueError("The length of training sentences must be equal to training labels.")
        if not len(val_sentences) == len(val_labels):
            raise ValueError("The length of validation sentences must be equal to validation labels.")

        train_token_ids, train_attention_masks = torch.tensor(
            get_TokenizedTextWithAttentionMask(train_sentences, self.tokenizer, self.max_length))
        val_token_ids, val_attention_masks = torch.tensor(
            get_TokenizedTextWithAttentionMask(val_sentences, self.tokenizer, self.max_length))

        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)

        train_data = TensorDataset(train_token_ids, train_attention_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        validation_data = TensorDataset(val_token_ids, val_attention_masks, val_labels)
        validation_sampler = SequentialSampler(validation_data)
        self.validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
        print(f'{len(train_sentences)} training and {len(val_sentences)} validation observations added successfully.')

    def train(self, epochs: int = 10, directory_path: str = 'models/', save_name: str = 'my_bert_classifier'):
        if self.train_dataloader is None or self.validation_dataloader is None:
            raise ValueError("First run 'add_data' method (function) to add data to the class.")
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        train_loss_set = []
        train_history = []
        valid_history = []
        best_val_accuracy = 0
        for epoch in range(epochs):
            print(f"starting epoch {epoch + 1} out of {epochs}")
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in tqdm(enumerate(self.train_dataloader)):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                train_loss_set.append(loss.item())
                loss.backward()
                self.optimizer.step()

                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("end of epoch {} - Train loss: {}".format(epoch, tr_loss / nb_tr_steps))
            train_history.append(tr_loss / nb_tr_steps)
            self.model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for batch in self.validation_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                    logits = output[0]

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
            Validation_Accuracy = (eval_accuracy / nb_eval_steps)
            valid_history.append(Validation_Accuracy)

            if Validation_Accuracy >= best_val_accuracy:
                p = directory_path + '/' + save_name + '.ckpt'
                torch.save(self.model.state_dict(), p)
                best_val_accuracy = Validation_Accuracy
                print(f'Model Saved at {p}')
        self.train_history = train_history
        self.valid_history = valid_history
        self.trained = True

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.loaded = True
        print(f"classifer loaded from {path}")

    def predict(self,
                test_sentences: List[str],
                batch_size: int = 32):
        if not self.trained and not self.loaded:
            raise ValueError("You must first either train (fine-tune) the model or load from a pretrained " +
                             "classifier and then predict.")
        warnings.warn('Adding data takes few seconds.')
        if not isinstance(test_sentences, list):
            raise ValueError("The test sentences and labels must be lists.")
        test_token_ids, test_attention_masks = torch.tensor(
            get_TokenizedTextWithAttentionMask(test_sentences, self.tokenizer, self.max_length))

        test_data = TensorDataset(test_token_ids, test_attention_masks)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        self.model.eval()
        all_logits = []

        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
                output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = output[0]

            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)
        return np.vstack(all_logits).argmax(1)


class YoutubeSearch:
    def __init__(self, DEVELOPER_KEY):
        self.YOUTUBE_API_SERVICE_NAME = "youtube"
        self.YOUTUBE_API_VERSION = "v3"
        self.DEVELOPER_KEY = DEVELOPER_KEY
        self.api = Api(api_key=self.DEVELOPER_KEY)  # initialise the Api

    def get_ids_by_keyword(self, keywordlist: List[str], count: int):
        """
        function to get a list of video ids associated with all words in keywordlist

        :param limit: limits the number of videos by request, we propose tpo take it on 50
        :param count: integer defining how many videos for a certain keyword shall be listed
        :param keywordlist: list[str] list of strings defining the keywords you want to search for
        :return: pandas dataframe: a dataframe consisting of 4 columns(video title, url, video id, publication date)

        """
        videos_id = []  # create an empty list for the video ids
        url = []  # create an empty list for the urls
        title = []  # create an empty list for the titels
        publication_date = []  # create an empty list for the published date
        description = []  # create an empty list for the description
        for keyword in tqdm(keywordlist):  # go through the keywordlist
            res = self.api.search_by_keywords(q=keyword, search_type=["video"], count=count,
                                              limit=50)  # define the search including limits and counts
            for item in res.items:  # fill in the different properties
                if item.snippet.title:
                    title.append(str(item.snippet.title))
                    url.append(f"https://youtube.com/watch?v={item.id.videoId}")
                    videos_id.append(item.id.videoId)
                    publication_date.append(item.snippet.publishedAt)
                    description.append(str(item.snippet.description))
        data = {'title': title, 'url': url, 'id': videos_id,
                'publication date': publication_date, 'description': description}  # summit the lists in an data object
        df = pd.DataFrame(data)  # turn the data object into a pandas data frame
        df['language'] = [detect_lang(x) for x in df['description'].values]
        return df  # return the dataframe

    def get_comments_by_id_list(self, video_id_list: List[str], count: int):
        """
        function to get the comments of the video by the video id list created before

        :param count: Number of comments we want to scrape maximally
        :param video_id_list: str list of strings containing the video ids we want to download the comments for
        :return: List[str]: list of strings encoded as utf-8 containing the comments of an video

        """
        comments = []
        for video_id in tqdm(video_id_list):  # go through the passed values of the video_id_list
            try:
                ct_by_video = self.api.get_comment_threads(video_id=video_id, count=count, limit=100)
                for item in ct_by_video.items:  # go through the items of the comment thread
                    if item.snippet.topLevelComment.snippet.textDisplay:
                        comment = str(item.snippet.topLevelComment.snippet.textDisplay)
                        try:
                            lang = detect(comment)
                        except:
                            lang = ''
                        comments.append([comment, lang, video_id])
            except:
                pass
        comments_df = pd.DataFrame(comments, columns=['comment', 'language', 'parent_video'])
        return comments_df

