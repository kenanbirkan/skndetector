from dermis.items import DermisItem
import datetime
import scrapy


class CoverSpider(scrapy.Spider):
    name = "dermisspider"
    start_urls = ["http://www.dermis.net/dermisroot/en/list/all/search.htm"]

    def parse(self, response):
        url_list = response.css("a")
        for item in url_list:
            link = item.xpath("@href").extract_first()
            if link and len(link) >0 and "diagnose" in link:
                yield scrapy.Request("http://www.dermis.net/dermisroot/" +link, self.parse_selected)  # TODO check extractfist




    def parse_selected(self,response):
        url = response.css(".diagnoseThumbs a")
        for href in url:
            yield scrapy.Request("http://www.dermis.net/dermisroot/" + href.xpath("@href").extract_first(), self.parse_page)  # TODO check extractfist




    def parse_page(self, response):
        # grab the URL of the cover image
        img = response.css("img")
        for elem in img:
            title = elem.xpath("@alt").extract_first()
            imageURL = "http://www.dermis.net" + elem.xpath("@src").extract_first()
            if "100px" in imageURL:
                imageURL=imageURL.replace("100px","550px")
            if title and "bilder" in imageURL:
                # yield the result
                try:
                    yield DermisItem(title=title,file_urls=[imageURL])
                except:
                    pass
            else:
                continue






