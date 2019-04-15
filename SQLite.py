# 导入头文件
import os 
import sqlite3

# 建立新的SQLite数据库文件并插入数据
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute('CREATE TABLE review_db'\
    ' (review TEXT, sentiment INTEGER, date TEXT)')
example1 = 'I love this movie'
c.execute('INSERT INTO review_db'\
    ' (review, sentiment, date) VALUES'\
    " (?, ?, DATETIME('now'))", (example1, 1))
example2 = 'I dislike this movie'
c.execute('INSERT INTO review_db'\
    ' (review, sentiment, date) VALUES'\
    " (?, ?, DATETIME('now'))", (example2, 0))
conn.commit()
conn.close()

# 通过SELECT语句查询数据
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute("SELECT * FROM review_db WHERE date"\
    " BETWEEN '2018-01-01 00:00:00' AND DATETIME('now')")
results = c.fetchall()
conn.close()
print(results)