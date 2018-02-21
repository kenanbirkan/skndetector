from globalskinatlas.items import GlobalskinatlasItem
import datetime
import scrapy


class CoverSpider(scrapy.Spider):
    name = "globalskinatlas"
    start_urls = ["http://www.globalskinatlas.com/diagindex.cfm"]

    def parse(self, response):
        url_list = response.css("a")
        for item in url_list:
            link = item.xpath("@href").extract_first()
            if link and len(link) >0 and "diagdetail" in link:
                yield scrapy.Request("http://www.globalskinatlas.com/" +link, self.parse_selected)  # TODO check extractfist




    def parse_selected(self,response):
        url_list = response.css("img")
        diagnosis = response.xpath('//strong/text()').extract_first()
        for item in url_list:
            link = item.xpath("@src").extract_first().replace("th","lg")

            if link and len(link) > 0 and "upload" in link:
                imageURL = link
                try:
                    yield GlobalskinatlasItem(title=diagnosis, file_urls=[imageURL])
                except:
                    pass










