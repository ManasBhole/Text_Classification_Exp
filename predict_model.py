from build_multi_channel_model import *
from create_train_test import clean_doc, load_doc
from keras.models import load_model

def predict_sentiment(review, vocab, tokenizer, length, model):
    	# clean review
	line = clean_doc(review)
	# encode and pad review
	padded = encode_text(tokenizer, [line], length)
	# predict sentiment
	yhat = model.predict([padded,padded,padded], verbose=0)
	# retrieve predicted percentage and label
	percent_pos = yhat[0,0]
	if  round(percent_pos) == 0:
		return (1-percent_pos), 'NEGATIVE'
	return percent_pos, 'POSITIVE'

def evaluate_model(model, trainX, trainLabels, testX, testLabels):
    _, acc = model.evaluate([trainX,trainX,trainX], array(trainLabels), verbose=0)
    print('Train Accuracy: %.2f' % (acc*100))
    # evaluate model on test dataset dataset
    _, acc = model.evaluate([testX,testX,testX], array(testLabels), verbose=0)
    print('Test Accuracy: %.2f' % (acc*100))
    
if __name__ == "__main__":
    testLines, testLabels = load_dataset('test.pkl')
    testX = encode_text(tokenizer, testLines, length)
    model = load_model('./model.h5')
    evaluate_model(model, trainX, trainLabels, testX, testLabels)
    text = 'This is a good movie'
    percent, sentiment = predict_sentiment(text, tokenizer.word_index.keys(), tokenizer, length, model)
    print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
    print()
    text = load_doc("C:/Users/asus/Deep_Learning_for_NLP/review_polarity/txt_sentoken/neg/cv003_12683.txt")
    percent, sentiment = predict_sentiment(text, tokenizer.word_index.keys(), tokenizer, length, model)
    print('Review from one of the documents: [%s...]\nSentiment: %s (%.3f%%)' % (text[3:20], sentiment, percent*100))