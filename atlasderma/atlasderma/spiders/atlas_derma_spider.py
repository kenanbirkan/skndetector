from atlasderma.items import AtlasdermaItem
import datetime
import scrapy


class CoverSpider(scrapy.Spider):
    name = "atlasdermaspider"
    start_urls = ["http://www.atlasdermatologico.com.br/search.jsf?q="]

    def parse(self, response):
        url_list = response.css("a")
        for item in url_list:
            link = item.xpath("@href").extract_first()
            if link and len(link) >0 and "disease" in link:
                yield scrapy.Request("http://www.atlasdermatologico.com.br/" +link, self.parse_selected)  # TODO check extractfist




    def parse_selected(self,response):
        url_list = response.css("img")
        for item in url_list:
            link = item.xpath("@src").extract_first().replace("&thumb=1","")
            title = item.xpath("@alt").extract_first()
            if link and len(link) > 0 and "imageId" in link:
                imageURL = "http://www.atlasdermatologico.com.br/" + link
                try:
                    yield AtlasdermaItem(title=title, file_urls=[imageURL])
                except:
                    pass










