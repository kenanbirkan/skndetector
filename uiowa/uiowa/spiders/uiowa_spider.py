from uiowa.items import UiowaItem
import datetime
import scrapy


class CoverSpider(scrapy.Spider):
    name = "uiowa"
    start_urls = ["https://medicine.uiowa.edu/dermatology/education/clinical-skin-disease-images"]

    def parse(self, response):
        url_list = response.css("a")
        for item in url_list:
            link = item.xpath("@href").extract_first()
            if link and len(link) >0 and "dermatology" in link:
                yield scrapy.Request("https:" +link, self.parse_selected)  # TODO check extractfist




    def parse_selected(self,response):
        # grab the URL of the cover image
        img = response.css("img")
        title = response.xpath('//title/text()').extract_first().split(" |")[0]
        for elem in img:
            imageURL = "https:"+ elem.xpath("@src").extract_first()
            if title:
                # yield the result
                yield UiowaItem(title=title, file_urls=[imageURL])
            else:
                continue








