import scrapy
from scrapy.crawler import CrawlerProcess

class CudaDocSpider(scrapy.Spider):
    name = "cuda_docs"
    start_urls = ['https://docs.nvidia.com/cuda/']

    def parse(self, response):
        content = response.css('body').getall()
        yield {'url': response.url, 'content': content}

        if response.meta.get('depth', 0) < 1:
            next_links = response.css('a::attr(href)').getall()
            for link in next_links:
                yield response.follow(link, self.parse, meta={'depth': response.meta.get('depth', 0) + 1})

process = CrawlerProcess(settings={
    'FEED_FORMAT': 'json',
    'FEED_URI': 'cuda_docs.json',
    'DEPTH_LIMIT': 1
})
process.crawl(CudaDocSpider)
process.start()
