import pandas as pd
import re
from underthesea import word_tokenize
from fastapi import FastAPI
from scipy.special import softmax
import numpy as np
import os
import uvicorn
from simpletransformers.classification import ClassificationModel, ClassificationArgs

np.set_printoptions(suppress=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# from underthesea import word_tokenize 
def preprocess(ls):
    df = pd.DataFrame({'text': ls})

    # Replace abbrv
    def replace_non_case(df):
        count = 0
        for pat, repl in abbreviation.items():
            df['text'] = df['text'].str.replace(pat, repl, case = False, regex = False)
            
            # Progress printing
            # if count % 10 == 0:
                # print(f"==== Preprocessing ({count}/{len(df)})")
            count += 1
        
        return df

    def sen_word_seg(sample):
        """ Viblo - phoBERT """
        
        splits = sample.strip().split('\n')
        text = ' '.join(splits)

        # return
        #text = segmenter.tokenize(text)
        #text = ' '.join([' '.join(x) for x in text])

        text = word_tokenize(text, format="text")
        return text

    def preprocess_v1(df):
        def clean_html(raw_html): #text
            cleanr = re.compile('<.*?>')
            cleantext = re.sub(cleanr, '', raw_html)
            return cleantext

         # Normalize unicode
        def normalize_unicode(txt): # text
            def loaddicchar():
                dic = {}
                char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
                    '|')
                charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
                    '|')
                for i in range(len(char1252)):
                    dic[char1252[i]] = charutf8[i]
                return dic

            def convert_unicode(txt):
                dicchar = loaddicchar()
                return re.sub(
                    r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
                    lambda x: dicchar[x.group()], txt)
            return convert_unicode(txt)
                


        # Remove hyperlinks in text
        def remove_urls(input_text): #text
            return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

        # Normalize Vietnamese
        # def normalize_vietnamese():
            # self.df['content'] = self.df['content'].progress_apply(lambda x: TTSnorm(x, lower = False, punc = True))



        df['text'] = df['text'].apply(clean_html)
        df['text'] = df['text'].apply(normalize_unicode)
        df['text'] = df['text'].apply(remove_urls)

        return df

    df = preprocess_v1(df)

    abbreviation = pd.read_csv('./resources/abbreviation.csv')
    abbreviation = abbreviation[['viet tat_x', 'mean']]
    abbreviation['len'] = abbreviation['mean'].apply(lambda x: len(x.split()))
    abbreviation = abbreviation[abbreviation['len'] > 2]

    abbreviation = dict(zip(abbreviation['viet tat_x'], abbreviation['mean']))

    abbreviation['csvn'] = 'Cộng sản Việt Nam'
    abbreviation['đcsvn'] = 'Đảng Cộng sản Việt Nam'
    abbreviation['xhcn'] = 'Xã hội Chủ nghĩa'
    abbreviation['cntb'] = 'Chủ nghĩa Tư bản'

    temp = df.copy()
    df = replace_non_case(temp)

    df = df['text'].apply(sen_word_seg)
    df = df.to_list()

    return df


if __name__ == "__main__":
    print("import libs...")

    model = ClassificationModel( "xlmroberta", "model", use_cuda = False)
    print("import model...")

    #ex = ["Bò đeo nơ ngốc nghếch"]
    # "Bò đeo nơ ngốc nghếch",
    # "Cộng sản là bọn bán nước hại dân",
    # "Nhân dân ta quyết tâm đánh bại dịch Covid, bảo vệ dân tộc", 
    # "Bọn cầm quyền tiếp tay cho Trung Cộng", 
    # "Chính quyền làm khó vậy là chết dân rồi"]

    #ex = preprocess(ex)

    #print("preproces...")

    #print(predict(ex))

    app = FastAPI()

    def predict(df):
        pred, prob = model.predict(df)
        result =  {'label': pred, 'probability': softmax(prob, axis = 1)}
        return result


    @app.get("/")
    async def read_root():
        return {"message": "Subscribe to @1littlecoder"}


    @app.get('/query/{msg}')
    def predict_query(msg: str):
        df = preprocess([msg])
        # print("preprocess: ", df) 
        return predict(df)


    #uvicorn.run(app, host="0.0.0.0", port=8000)
   
