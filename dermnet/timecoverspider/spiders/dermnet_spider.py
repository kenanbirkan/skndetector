from dermnet.items import DermItem
import datetime
import scrapy


class CoverSpider(scrapy.Spider):
    name = "dermnet"
    start_urls = ["http://www.dermnet.com/dermatology-pictures-skin-disease-pictures"]

    def parse(self, response):
        url_list = response.css("a")
        for item in url_list:
            link = item.xpath("@href").extract_first()
            if link and len(link) >0 and "images" in link:
                yield scrapy.Request("http://www.dermnet.com/" +link, self.parse_selected)  # TODO check extractfist




    def parse_selected(self,response):
        url = response.css(".thumbnails a")
        for href in url:
            yield scrapy.Request("http://www.dermnet.com/images/" + href.xpath("@href").extract_first(), self.parse_page)  # TODO check extractfist

        # extract the 'Next' link from the pagination, load it, and
        # parse it
        next = response.css("div.pagination").xpath("a[contains(., 'Next')]")
        if next and len(next)>0:
            yield scrapy.Request("http://www.dermnet.com/" + next.xpath("@href").extract_first(), self.parse_selected)




    def parse_page(self, response):
        # grab the URL of the cover image
        img = response.css("img")
        for elem in img:
            title = elem.xpath("@title").extract_first()
            imageURL = elem.xpath("@src").extract_first()
            if title:
                # yield the result
                yield DermItem(title=title,file_urls=[imageURL])
            else:
                continue






