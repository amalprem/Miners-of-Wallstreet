news_output = {"data":"Stock news are on the positive side of spectrum for today","date":"2023-04-17"}

response = {
    'news_output': news_output
}
newsScore = 1 if "positive" in response['news_output']['data'] else 0
print(newsScore)