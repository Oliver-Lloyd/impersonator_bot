import tweepy
from bot.functions import *
from time import sleep

if __name__ == '__main__':

    # Set up API connection
    consumer_key = open('twitter/twitter_keys/consumer_key.txt').read()
    consumer_secret = open('twitter/twitter_keys/consumer_secret.txt').read()
    access_token = open('twitter/twitter_keys/access_token.txt').read()
    access_token_secret = open('twitter/twitter_keys/access_token_secret.txt').read()
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # Checking API connection
    try:
        public_tweets = api.home_timeline()
        print('Successfully connected to Twitter.')
    except tweepy.error.TweepError:
        print('Tweepy Error. Please check Twitter API keys.')
        quit()

    last_mention = None
    while True:
        # Get mentions that haven't been processed
        if last_mention is None:
            mentions = api.mentions_timeline()
        else:
            mentions = api.mentions_timeline(since_id=last_mention)

        for mention in mentions:
            last_mention = mention
            input_text = None
            split_tweet = mention.text.split('"')

            if len(split_tweet) == 3:  # check that the tweet is correctly formatted, otherwise ignore
                request = split_tweet[1]
                response = generate(request, 25)
                print(response)
                api.update_status(response, in_reply_to_status_id=mention.id)
