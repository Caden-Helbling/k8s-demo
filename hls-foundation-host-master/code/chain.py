import copy
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import posixpath
import requests
import time

from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import HumanMessage, SystemMessage

from opencage.geocoder import OpenCageGeocode

from tqdm import tqdm
from typing import Tuple, Optional
from urllib.parse import urljoin

THRESHOLD = 0.17


class DialogParser:
    PARSER_PROMPT = f"""
        You are a tool that converts incoming natural text to list of parameters that can be used for an API call.
        There are 3 parameter values (named entitites) that need to be understood and parsed from the text.

        Parameters:
            Area: This is the location name.
            Date: This is the date in the format YYYY-MM-DD for which forest data has to be fetched. This can be a single value or list of data based on user input texts.
            Event Type: This is the number of days for which data is fetched. Its value strictly ranges from 1 to 10.
            Error: This is the message filled if area, event, or date are not discernable from the message.

        Your task is to parse input texts and give the values of all these 3 parameters, the area, date, and event type.

        Following instructions are used to show how to parse date values:
            Assume today's date is {time.strftime('%Y-%m-%d')}.
            If user specifies exactly N years ago, use today's date and compute the date as today's date - N and provide single value date. Don't generate list in this case. Only provide single date.
            If date is not provided directly, you can get relative date based on today's date. For example, if the users says 'data from last year', you have to use today's date and give the date of last year. And generate list of YYYY-MM-DD values starting from January of that year to ending at December of that year.
            If user specifies a specific month and year, just use first day of that month and only that year and give single value date. Keep the day range value as it is.
        If you cannot determine the location or event type or date, add message to the "error" key in the output.


        Strictly provide area as name of a location anywhere in the world. Do not accept any name as the location name.

        Event Type should be either "burn_scars" or "flood". If it is anything else, don't include it.

        Strictly output json only. Remove any additional notes and comments.

        If you cannot prepare the json, respond with a json in the following format:
            'error': <reason for error>
    """
    RESTRUCTURE_PROMPT = f"""
        Give me the data strictly in json format as:
            'area': <area>
            'date': <date>
            'event_type': <event type>
            'error': <error message>

        Don't output anything else. Make sure keys and values are in double quotes. Not single quotes.
        Remove any additional notes and comments. Don't add any extra comments and texts anywhere.

        If date is greater than today's date which is {time.strftime('%Y-%m-%d')}, set the date to today's date.

        Area should strictly be a name of a location.

        Event Type should be either "burn_scars" or "flood". If it is anything else, don't include it.

        If you cannot prepare the json, respond with a json in the following format:
            'error': <reason for error>
    """

    SUMMARIZE_PROMPT = """
        You are a summarizer. You will be provided with a json with the following format.

        Your task is to explain the json in simple language. Do not add any other insight other than the provided information in plain language.
    """

    SUMMARIZE_RESPONSE_PROMPT = """
        Reason with yourself to make sure you have followed all rules to explain the query. Then re-write the query following all previous rules.
    """

    def __init__(
        self,
        parser_prompt: Optional[str] = None,
        restructure_prompt: Optional[str] = None,
    ) -> None:
        self.llm = ChatOpenAI(temperature=0.1, cache=False)
        self.parser_prompt = parser_prompt or DialogParser.PARSER_PROMPT
        self.restructure_prompt = restructure_prompt = DialogParser.RESTRUCTURE_PROMPT

    @property
    def today_str(self) -> str:
        return time.strftime("%Y-%m-%d")

    def run_model(self, messages: Tuple[SystemMessage, HumanMessage]) -> str:
        output = self.llm._generate(messages)
        # Extract and return the generated text from the model output
        return output.generations[0].text.strip()

    def parse(self, text) -> dict:
        prompt_message = SystemMessage(content=self.parser_prompt.strip())
        user_message = HumanMessage(content=text.strip())
        restructure_message = SystemMessage(content=self.restructure_prompt.strip())

        message = (prompt_message, user_message, restructure_message)

        res = self.run_model(message)
        try:
            res = json.loads(res)
        except:
            res = json.dumps({"error": res})
        if area := res.get('area'):
            res['bounding_box'] = self.geocode(area)
        return res

    def geocode(self, text: str) -> str:
        """Geocode a query (location, region, or landmark)"""
        opencage_geocoder = OpenCageGeocode(os.environ["OPENCAGE_API_KEY"])
        response = opencage_geocoder.geocode(text, no_annotations="1")
        if response:
            bounds = response[0]["geometry"]

            # convert to bbox
            return [
                bounds["lng"] - THRESHOLD,
                bounds["lat"] - THRESHOLD,
                bounds["lng"] + THRESHOLD,
                bounds["lat"] + THRESHOLD,
            ]
