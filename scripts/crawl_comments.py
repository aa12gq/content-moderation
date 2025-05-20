import os
import csv
import time
import random
import requests
from typing import List, Dict
from bs4 import BeautifulSoup

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
OUTPUT_FILE = os.path.join(DATA_DIR, 'raw_comments.csv')

os.makedirs(DATA_DIR, exist_ok=True)

PLATFORMS = ['douban', 'zhihu', 'bilibili', 'weibo']

def get_headers():
    """获取随机User-Agent"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    ]
    return {"User-Agent": random.choice(user_agents)}

# 获取豆瓣热门电影ID
def get_douban_hot_movies(max_count=10) -> List[str]:
    print("正在获取豆瓣热门电影...")
    movie_ids = []
    try:
        url = "https://movie.douban.com/j/search_subjects?type=movie&tag=热门&sort=recommend&page_limit=50&page_start=0"
        resp = requests.get(url, headers=get_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get('subjects', []):
                movie_id = item.get('id')
                if movie_id:
                    movie_ids.append(movie_id)
                if len(movie_ids) >= max_count:
                    break
    except Exception as e:
        print(f"获取豆瓣热门电影异常: {e}")
    return movie_ids[:max_count]

# 获取知乎热门问题ID
def get_zhihu_hot_questions(max_count=10) -> List[str]:
    print("正在获取知乎热门问题...")
    question_ids = []
    try:
        url = "https://www.zhihu.com/hot"
        resp = requests.get(url, headers=get_headers(), timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/question/' in href:
                    qid = href.split('/question/')[1].split('/')[0]
                    if qid.isdigit() and qid not in question_ids:
                        question_ids.append(qid)
                    if len(question_ids) >= max_count:
                        break
    except Exception as e:
        print(f"获取知乎热门问题异常: {e}")
    return question_ids[:max_count]

# 获取B站热门视频ID
def get_bilibili_hot_videos(max_count=10) -> List[str]:
    print("正在获取B站热门视频...")
    video_ids = []
    try:
        url = "https://api.bilibili.com/x/web-interface/popular?ps=50&pn=1"
        resp = requests.get(url, headers=get_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get('data', {}).get('list', []):
                video_id = str(item.get('aid'))
                if video_id:
                    video_ids.append(video_id)
                if len(video_ids) >= max_count:
                    break
    except Exception as e:
        print(f"获取B站热门视频异常: {e}")
    return video_ids[:max_count]

# 获取微博热门话题ID
def get_weibo_hot_topics(max_count=10) -> List[str]:
    print("正在获取微博热门话题...")
    topic_ids = []
    try:
        url = "https://weibo.com/ajax/side/hotSearch"
        resp = requests.get(url, headers=get_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get('data', {}).get('realtime', []):
                topic_id = item.get('mid')
                if topic_id:
                    topic_ids.append(topic_id)
                if len(topic_ids) >= max_count:
                    break
    except Exception as e:
        print(f"获取微博热门话题异常: {e}")
    return topic_ids[:max_count]

# 豆瓣短评采集
def crawl_douban_short_comments(movie_id: str, max_count=100) -> List[str]:
    print(f"采集豆瓣短评: {movie_id}")
    comments = []
    start = 0
    while len(comments) < max_count:
        url = f"https://movie.douban.com/subject/{movie_id}/comments?start={start}&limit=20&status=P&sort=new_score"
        try:
            resp = requests.get(url, headers=get_headers(), timeout=10)
            if resp.status_code != 200:
                break
            html = resp.text
            import re
            items = re.findall(r'<span class="short">(.*?)</span>', html, re.S)
            if not items:
                break
            comments.extend([c.strip() for c in items])
            start += 20
            time.sleep(random.uniform(1, 2))
        except Exception as e:
            print(f"豆瓣采集异常: {e}")
            break
    return comments[:max_count]

# 知乎回答采集
def crawl_zhihu_answers(question_id: str, max_count=100) -> List[str]:
    print(f"采集知乎回答: {question_id}")
    comments = []
    offset = 0
    while len(comments) < max_count:
        url = f"https://www.zhihu.com/api/v4/questions/{question_id}/answers?limit=20&offset={offset}"
        try:
            resp = requests.get(url, headers=get_headers(), timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
            for ans in data.get('data', []):
                txt = ans.get('excerpt', '').strip()
                if txt:
                    comments.append(txt)
            if not data.get('paging', {}).get('is_end', True):
                offset += 20
            else:
                break
            time.sleep(random.uniform(1, 2))
        except Exception as e:
            print(f"知乎采集异常: {e}")
            break
    return comments[:max_count]

# B站弹幕采集
def crawl_bilibili_danmu(cid: str, max_count=100) -> List[str]:
    print(f"采集B站弹幕: {cid}")
    url = f"https://api.bilibili.com/x/v1/dm/list.so?oid={cid}"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=10)
        if resp.status_code != 200:
            return []
        from xml.etree import ElementTree
        xml = ElementTree.fromstring(resp.content)
        comments = [d.text for d in xml.findall('.//d') if d.text]
        return comments[:max_count]
    except Exception as e:
        print(f"B站采集异常: {e}")
        return []

# 微博评论采集
def crawl_weibo_comments(weibo_id: str, max_count=100) -> List[str]:
    print(f"采集微博评论: {weibo_id}")
    comments = []
    max_id = ''
    while len(comments) < max_count:
        url = f"https://m.weibo.cn/comments/hotflow?id={weibo_id}&mid={weibo_id}&max_id={max_id}&max_id_type=0"
        try:
            resp = requests.get(url, headers=get_headers(), timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
            for c in data.get('data', {}).get('data', []):
                txt = c.get('text', '').strip()
                import re
                txt = re.sub('<.*?>', '', txt)
                if txt:
                    comments.append(txt)
            max_id = data.get('data', {}).get('max_id', '')
            if not max_id:
                break
            time.sleep(random.uniform(1, 2))
        except Exception as e:
            print(f"微博采集异常: {e}")
            break
    return comments[:max_count]

def save_comments(platform: str, comments: List[str]):
    with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for c in comments:
            writer.writerow([platform, c])

def main():
    # 自动获取热门内容ID
    print("开始获取热门内容ID...")
    douban_ids = get_douban_hot_movies(5)  # 获取5个热门电影
    zhihu_ids = get_zhihu_hot_questions(5)  # 获取5个热门问题
    bilibili_ids = get_bilibili_hot_videos(5)  # 获取5个热门视频
    weibo_ids = get_weibo_hot_topics(5)  # 获取5个热门话题

    print(f"\n获取到的ID:")
    print(f"豆瓣电影: {douban_ids}")
    print(f"知乎问题: {zhihu_ids}")
    print(f"B站视频: {bilibili_ids}")
    print(f"微博话题: {weibo_ids}\n")

    # 清空旧文件
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    print(f"采集结果将保存到: {OUTPUT_FILE}")

    # 采集评论
    for did in douban_ids:
        save_comments('douban', crawl_douban_short_comments(did, 100))
    for zid in zhihu_ids:
        save_comments('zhihu', crawl_zhihu_answers(zid, 100))
    for cid in bilibili_ids:
        save_comments('bilibili', crawl_bilibili_danmu(cid, 100))
    for wid in weibo_ids:
        save_comments('weibo', crawl_weibo_comments(wid, 100))

    print("\n采集完成！请用Excel或doccano进行人工标注。")

if __name__ == '__main__':
    main() 