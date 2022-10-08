#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import re
from typing import List, Union, Dict


class CommonRegex(object):

    regexes = {
        "date": re.compile(
            r"(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}|\d{2,4}[-\./][0-3]?\d[-\./][0-3]?\d",
            re.IGNORECASE,
        ),
        "time": re.compile(
            r"\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?", re.IGNORECASE
        ),
        "phone_number_US": re.compile(
            r"((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))"
        ),
        "phone_number_US_with_ext": re.compile(
            r"((?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?(?:[2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?(?:[0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(?:\d+)?))",
            re.IGNORECASE,
        ),
        "link": re.compile(
            r'(?i)((?:https?://|www\d{0,3}[.])?[a-z0-9.\-]+[.](?:(?:international)|(?:construction)|(?:contractors)|(?:enterprises)|(?:photography)|(?:immobilien)|(?:management)|(?:technology)|(?:directory)|(?:education)|(?:equipment)|(?:institute)|(?:marketing)|(?:solutions)|(?:builders)|(?:clothing)|(?:computer)|(?:democrat)|(?:diamonds)|(?:graphics)|(?:holdings)|(?:lighting)|(?:plumbing)|(?:training)|(?:ventures)|(?:academy)|(?:careers)|(?:company)|(?:domains)|(?:florist)|(?:gallery)|(?:guitars)|(?:holiday)|(?:kitchen)|(?:recipes)|(?:shiksha)|(?:singles)|(?:support)|(?:systems)|(?:agency)|(?:berlin)|(?:camera)|(?:center)|(?:coffee)|(?:estate)|(?:kaufen)|(?:luxury)|(?:monash)|(?:museum)|(?:photos)|(?:repair)|(?:social)|(?:tattoo)|(?:travel)|(?:viajes)|(?:voyage)|(?:build)|(?:cheap)|(?:codes)|(?:dance)|(?:email)|(?:glass)|(?:house)|(?:ninja)|(?:photo)|(?:shoes)|(?:solar)|(?:today)|(?:aero)|(?:arpa)|(?:asia)|(?:bike)|(?:buzz)|(?:camp)|(?:club)|(?:coop)|(?:farm)|(?:gift)|(?:guru)|(?:info)|(?:jobs)|(?:kiwi)|(?:land)|(?:limo)|(?:link)|(?:menu)|(?:mobi)|(?:moda)|(?:name)|(?:pics)|(?:pink)|(?:post)|(?:rich)|(?:ruhr)|(?:sexy)|(?:tips)|(?:wang)|(?:wien)|(?:zone)|(?:biz)|(?:cab)|(?:cat)|(?:ceo)|(?:com)|(?:edu)|(?:gov)|(?:int)|(?:mil)|(?:net)|(?:onl)|(?:org)|(?:pro)|(?:red)|(?:tel)|(?:uno)|(?:xxx)|(?:ac)|(?:ad)|(?:ae)|(?:af)|(?:ag)|(?:ai)|(?:al)|(?:am)|(?:an)|(?:ao)|(?:aq)|(?:ar)|(?:as)|(?:at)|(?:au)|(?:aw)|(?:ax)|(?:az)|(?:ba)|(?:bb)|(?:bd)|(?:be)|(?:bf)|(?:bg)|(?:bh)|(?:bi)|(?:bj)|(?:bm)|(?:bn)|(?:bo)|(?:br)|(?:bs)|(?:bt)|(?:bv)|(?:bw)|(?:by)|(?:bz)|(?:ca)|(?:cc)|(?:cd)|(?:cf)|(?:cg)|(?:ch)|(?:ci)|(?:ck)|(?:cl)|(?:cm)|(?:cn)|(?:co)|(?:cr)|(?:cu)|(?:cv)|(?:cw)|(?:cx)|(?:cy)|(?:cz)|(?:de)|(?:dj)|(?:dk)|(?:dm)|(?:do)|(?:dz)|(?:ec)|(?:ee)|(?:eg)|(?:er)|(?:es)|(?:et)|(?:eu)|(?:fi)|(?:fj)|(?:fk)|(?:fm)|(?:fo)|(?:fr)|(?:ga)|(?:gb)|(?:gd)|(?:ge)|(?:gf)|(?:gg)|(?:gh)|(?:gi)|(?:gl)|(?:gm)|(?:gn)|(?:gp)|(?:gq)|(?:gr)|(?:gs)|(?:gt)|(?:gu)|(?:gw)|(?:gy)|(?:hk)|(?:hm)|(?:hn)|(?:hr)|(?:ht)|(?:hu)|(?:id)|(?:ie)|(?:il)|(?:im)|(?:in)|(?:io)|(?:iq)|(?:ir)|(?:is)|(?:it)|(?:je)|(?:jm)|(?:jo)|(?:jp)|(?:ke)|(?:kg)|(?:kh)|(?:ki)|(?:km)|(?:kn)|(?:kp)|(?:kr)|(?:kw)|(?:ky)|(?:kz)|(?:la)|(?:lb)|(?:lc)|(?:li)|(?:lk)|(?:lr)|(?:ls)|(?:lt)|(?:lu)|(?:lv)|(?:ly)|(?:ma)|(?:mc)|(?:md)|(?:me)|(?:mg)|(?:mh)|(?:mk)|(?:ml)|(?:mm)|(?:mn)|(?:mo)|(?:mp)|(?:mq)|(?:mr)|(?:ms)|(?:mt)|(?:mu)|(?:mv)|(?:mw)|(?:mx)|(?:my)|(?:mz)|(?:na)|(?:nc)|(?:ne)|(?:nf)|(?:ng)|(?:ni)|(?:nl)|(?:no)|(?:np)|(?:nr)|(?:nu)|(?:nz)|(?:om)|(?:pa)|(?:pe)|(?:pf)|(?:pg)|(?:ph)|(?:pk)|(?:pl)|(?:pm)|(?:pn)|(?:pr)|(?:ps)|(?:pt)|(?:pw)|(?:py)|(?:qa)|(?:re)|(?:ro)|(?:rs)|(?:ru)|(?:rw)|(?:sa)|(?:sb)|(?:sc)|(?:sd)|(?:se)|(?:sg)|(?:sh)|(?:si)|(?:sj)|(?:sk)|(?:sl)|(?:sm)|(?:sn)|(?:so)|(?:sr)|(?:st)|(?:su)|(?:sv)|(?:sx)|(?:sy)|(?:sz)|(?:tc)|(?:td)|(?:tf)|(?:tg)|(?:th)|(?:tj)|(?:tk)|(?:tl)|(?:tm)|(?:tn)|(?:to)|(?:tp)|(?:tr)|(?:tt)|(?:tv)|(?:tw)|(?:tz)|(?:ua)|(?:ug)|(?:uk)|(?:us)|(?:uy)|(?:uz)|(?:va)|(?:vc)|(?:ve)|(?:vg)|(?:vi)|(?:vn)|(?:vu)|(?:wf)|(?:ws)|(?:ye)|(?:yt)|(?:za)|(?:zm)|(?:zw))(?:/[^\s()<>]+[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019])?)',
            re.IGNORECASE,
        ),
        "email": re.compile(
            r"([a-z0-9!#$%&\'*+\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)",
            re.IGNORECASE,
        ),
        "ip": re.compile(
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
            re.IGNORECASE,
        ),
        "ipv6": re.compile(
            r"\s*(?!.*::.*::)(?:(?!:)|:(?=:))(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)){6}(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)[0-9a-f]{0,4}(?:(?<=::)|(?<!:)|(?<=:)(?<!::):)|(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)){3})\s*",
            re.VERBOSE | re.IGNORECASE | re.DOTALL,
        ),
        "price": re.compile(
            r"[$]\s?[+-]?[0-9]{1,3}(?:(?:,?[0-9]{3}))*(?:\.[0-9]{1,2})?"
        ),
        "credit_card": re.compile(r"((?:(?:\d{4}[- ]?){3}\d{4}|\d{15,16}))(?![\d])"),
        "address": re.compile(
            r"\d{1,5} [\w\s]{1,30}(?:street|st|crescent|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)",
            re.IGNORECASE,
        ),
        "zip_code": re.compile(r"\b\d{5}(?:[-\s]\d{4})?\b"),
        "po_box": re.compile(r"P\.? ?O\.? Box \d+", re.IGNORECASE),
        "ssn": re.compile(r"(?!666|000|9\d{2})\d{3}[- ](?!00)\d{2}[- ](?!0{4})\d{4}"),
        "address_with_zip": re.compile(
            r"\d{1,5} [\w\s]{1,30}(?:street|st(?:\s|\.)+|avenue|ave(?:\s|\.)+|road|rd(?:\s|\.)+|highway|hwy(?:\s|\.)+|square|sq(?:\s|\.)+|trail|trl(?:\s|\.)+|drive|dr(?:\s|\.)+|court|ct(?:\s|\.)+|park|parkway|pkwy(?:\s|\.)+|circle|cir(?:\s|\.)+|boulevard|blvd(?:\s|\.)+|island|port|view|parkways)(?:suite\s?\d+|apt\.?\s?\d+|ste\.?\s?\d+)?[\w\s,]{1,30}\d{5}\W?(?=\s|$)",
            re.IGNORECASE,
        ),
    }

    class regex:
        def __init__(self, obj, regex):
            self.obj = obj
            self.regex = regex

        def __call__(self, *args):
            def regex_method(text=None):
                return [x.strip() for x in self.regex.findall(text or self.obj.text)]

            return regex_method

    def __init__(self, text=""):
        self.text = text

        for k, v in CommonRegex.regexes.items():
            setattr(self, k, CommonRegex.regex(self, v)(self))

        if text:
            for key in CommonRegex.regexes.keys():
                method = getattr(self, key)
                setattr(self, key, method())


class CommonRegexMixin(object):

    redact_map = {key: "[" + key.upper() + "]" for key in CommonRegex.regexes.keys()}

    def __init__(self):
        self.parsed = CommonRegex(self.string)

    @property
    def date(self):
        return self.parsed.date

    @property
    def time(self):
        return self.parsed.time

    @property
    def phone_number_US(self):
        return self.parsed.phone_number_US + self.parsed.phone_number_US_with_ext

    @property
    def link(self):
        return self.parsed.link

    @property
    def email(self):
        return self.parsed.email

    @property
    def ip(self):
        return self.parsed.ip + self.parsed.ipv6

    @property
    def price(self):
        return self.parsed.price

    @property
    def credit_card(self):
        return self.parsed.credit_card

    @property
    def address(self):
        return self.parsed.address + self.parsed.address_with_zip

    @property
    def zip_code(self):
        return self.parsed.zip_code

    @property
    def ssn(self):
        return self.parsed.ssn

    def redact(self, fields: Union[List[str], Dict[str, str]]) -> str:
        """Remove personal information in a string.
        For example, "Jane's phone number is 123-456-7890" is turned into "Jane's phone number is [phone_number_US]."

        Parameters
        ----------
        fields: (list(str) | dict)
            either a list of fields to redact, e.g. ['email', 'phone_number_US'], in which case the redacted text is replaced
            with capitalized word like [EMAIL] or [PHONE_NUMBER_US_WITH_EXT], or a dictionary where key is a field to redact and value
            is the replacement text, e.g., {'email': 'HIDDEN_EMAIL'}.

        Returns
        -------
        str
            redacted string
        """
        if isinstance(fields, list):
            redact_map = {
                field: self.redact_map[field]
                for field in fields
                if field in self.redact_map
            }
        else:
            redact_map = {
                k: "[" + v + "]" for k, v in fields.items() if k in self.redact_map
            }
        _string = self.string
        for k, v in redact_map.items():
            _string = re.sub(CommonRegex.regexes[k], v, _string)
        return _string
