from hellenic.items import HellenicItem
import datetime
import scrapy


class CoverSpider(scrapy.Spider):
    name = "hellenic"
    start_urls = ["http://www.hellenicdermatlas.com/en/search/browse/"]

    def parse(self, response):
        url_list = response.css("a")
        for item in url_list:
            link = item.xpath("@href").extract_first()
            if link and len(link) >0 and "PICTURES" in link:
                yield scrapy.Request("http://www.hellenicdermatlas.com/" +link, self.parse_selected)  # TODO check extractfist




    def parse_selected(self,response):
        import time
        time.sleep(5)
        url = response.css("a")
        for href in url:
            p_url =  href.xpath("@href").extract_first()
            if p_url and "viewpicture" in p_url:
                yield scrapy.Request("http://www.hellenicdermatlas.com/" + p_url, self.parse_page)  # TODO check extractfist

        # extract the 'Next' link from the pagination, load it, and
        # parse it
        next = response.css("div.paging").xpath("a[contains(., 'Next')]")
        if next and len(next)>0:
            yield scrapy.Request("http://www.hellenicdermatlas.com/" + next.xpath("@href").extract_first(), self.parse_selected)




    def parse_page(self, response):
        # grab the URL of the cover image
        img = response.css("img")
        for elem in img:
            title = elem.xpath("@title").extract_first()
            imageURL = "http://www.hellenicdermatlas.com" + elem.xpath("@src").extract_first()
            if title and "photos" in imageURL:
                # yield the result
                yield HellenicItem(title=title,file_urls=[imageURL])
            else:
                continue






