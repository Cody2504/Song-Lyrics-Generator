# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy



from scrapy import Item, Field

class SongItem(Item):
    title = Field()
    content = Field()
    url = Field()