# import the necessary packages
import scrapy


class DermItem(scrapy.Item):
    title = scrapy.Field()
    file_urls = scrapy.Field()
    files = scrapy.Field()
