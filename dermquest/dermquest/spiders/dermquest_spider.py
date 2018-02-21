from dermquest.items import DermquestItem
import datetime
import scrapy
import traceback
from scrapy.http import FormRequest
import json
class CoverSpider(scrapy.Spider):
    name = "dermquest"
    start_urls = ["https://www.dermquest.com/image-library/image-search/"]


    def parse(self, response):
        for i in range(109487, 110239):
            for j in range(1,6):
                link = "https://www.dermquest.com/Services/imageData.ashx?diagnosis=" + str(
                    i) + "&perPage=500&page="+str(j)
                yield FormRequest(url=link,
                                  formdata={
                                      '__VIEWSTATE': response.css('input#__VIEWSTATE::attr(value)').extract_first(),
                                                                    },
                        callback=self.parse_selected)




    def parse_selected(self,response):
        try:
            json_response = json.loads(response.text)
            for item in json_response["Results"]:
                link = "https://www.dermquest.com/image-library/image/" + item["AssetId"]
                yield scrapy.Request(link, self.parse_page)  # TODO check extractfist
        except:
            pass




    def parse_page(self, response):
        # grab the URL of the cover image
        img = response.css("img")
        for elem in img:
            title = elem.xpath("@alt").extract_first()
            imageURL = "https://www.dermquest.com" + elem.xpath("@src").extract_first()
            if imageURL and "medium" in imageURL:
                # yield the result
                yield DermquestItem(title=title,file_urls=[imageURL])
            else:
                continue





