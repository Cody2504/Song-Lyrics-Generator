import scrapy
from scrapy.exceptions import DropItem
import time

class SongSpider(scrapy.Spider):
    name = 'song_scrape'
    custom_settings = {
        'CONCURRENT_REQUESTS': 16,
        'DOWNLOAD_DELAY': 1,
        'RANDOMIZE_DOWNLOAD_DELAY': True
    }

    def start_requests(self):
        base_url = 'https://www.lyrics.com/artist/Taylor-Swift/816977'
        yield scrapy.Request(url=base_url, callback=self.parse_list_page)

    def parse_list_page(self, response):
        base_url = 'https://www.lyrics.com'
        Song_links = response.xpath("//td[@class='tal qx']/strong/a/@href").getall()
        print(Song_links)

        for link in Song_links:
            full_url = base_url + link  
            yield scrapy.Request(url=full_url, callback=self.parse_song)


    def parse_song(self, response):
        content_xpath = "/html/body/div[1]/div/div[2]/div[1]/div[1]/div[1]/div[1]/pre"
        content_elements = response.xpath(content_xpath)
        
        content = []
        for element in content_elements:
            # Extract text while removing i and b tags
            text = ''.join(element.xpath('.//text()').getall()).strip()
            if text:
                content.append(text)
        
        
        yield {
            'title': response.xpath('//h1[@class=";yric-title"]/text()').get('').strip(),
            'content': '\n'.join(content),
            'url': response.url
        }