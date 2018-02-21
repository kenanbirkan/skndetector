from dermnetz.items import DermnetzItem
import datetime
import scrapy


class CoverSpider(scrapy.Spider):
    name = "dermnetz"
    start_urls = ["https://www.dermnetnz.org/topics/"]

    def parse(self, response):
        url_list = response.css("a")
        for item in url_list:
            link = item.xpath("@href").extract_first()
            if link and len(link) >0 and "topics" in link:
                yield scrapy.Request("https://www.dermnetnz.org" +link, self.parse_selected)  # TODO check extractfist




    def parse_selected(self,response):
        url = response.css("a")
        for href in url:
            link = href.xpath("@href").extract_first()
            if link and "imagedetail" in link:
                yield scrapy.Request("https://www.dermnetnz.org/" + link, self.parse_page)  # TODO check extractfist





    def parse_page(self, response):
        # grab the URL of the cover image
        img = response.css("img")
        for elem in img:
            title = elem.xpath("@alt").extract_first()
            imageURL = "https://www.dermnetnz.org" + elem.xpath("@src").extract_first()
            if imageURL and "assets" in imageURL:
                # yield the result
                yield DermnetzItem(title=title,file_urls=[imageURL])
            else:
                continue





