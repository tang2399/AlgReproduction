from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 测试文本
corpus = [
    'dog cat fish.',
    'dog cat cat'
]

# 词频统计
vectorizer = CountVectorizer()
# 将计算的结果转换出来
X = vectorizer.fit_transform(corpus)
print(X.toarray())

# 词加权计算
vectorizer = TfidfVectorizer()
# 将计算的结果转换出来
X = vectorizer.fit_transform(corpus)
print(X.toarray())