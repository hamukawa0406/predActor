from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={"root_dir": "gohan"})
crawler.crawl(keyword="孫悟飯", max_num=100)