import numpy as np
from copy import deepcopy


class data_preprocessing:
    def __init__(self, instance_path):
        self.instance_path = instance_path

        self.info, self.flights = self.read_file(f_name=self.instance_path)
        self.number_of_areas, self.starting_airport = (
            int(self.info[0][0]),
            self.info[0][1],
        )

        self.flights_by_day_dict = self.flights_by_day(flight_list=self.flights)

        self.flights_by_day_dict = self.remove_duplicate(
            flights_by_day=self.flights_by_day_dict
        )

        self.list_days = [k for k in range(1, self.number_of_areas)]

        self.airports_by_area = self.get_airports_by_areas()
        self.area_to_explore = self.which_area_to_explore(
            airports_by_area=self.airports_by_area
        )
        self.area_by_airport = self.invert_dict(original_dict=self.airports_by_area)

        self.starting_area = self.associated_area_to_airport(
            airport=self.starting_airport
        )
        self.list_airports = self.get_list_of_airports()
        self.list_areas = list(self.airports_by_area.keys())
        self.areas_connections_by_day = (
            self.possible_flights_from_zone_to_zone_specific_day()
        )

    def read_file(self, f_name):
        dist = []
        line_nu = -1
        with open(f_name) as infile:
            for line in infile:
                line_nu += 1
                if line_nu == 0:
                    index = int(line.split()[0]) * 2 + 1
                if line_nu >= index:
                    temp = line.split()
                    temp[2] = int(temp[2])
                    temp[3] = int(temp[3])
                    dist.append(temp)
                else:
                    dist.append(line.split())
            info = dist[: int(dist[0][0]) * 2 + 1]
            flights = dist[int(dist[0][0]) * 2 + 1 :]
        return info, flights

    def flights_by_day(self, flight_list):
        # Create an empty dictionary to hold flights organized by day
        flights_by_day = {}

        # Iterate over each flight in the input list
        for flight in flight_list:
            # Extract the day from the flight entry
            day = flight[2]

            # Create a flight entry without the day
            flight_without_day = flight[:2] + flight[3:]

            # Add the flight to the corresponding day in the dictionary
            if day not in flights_by_day:
                flights_by_day[day] = []
            flights_by_day[day].append(flight_without_day)

        return flights_by_day

    def flights_from_airport(self, flights_by_day, from_airport, considered_day):
        flights_from_airport = []
        for day, flights in flights_by_day.items():
            if day == considered_day:
                for flight in flights:
                    if flight[0] == from_airport:
                        flights_from_airport.append(flight)
                return flights_from_airport
            else:
                return None

    def invert_dict(self, original_dict):
        inverted_dict = {}
        for key, value_list in original_dict.items():
            for value in value_list:
                if value in inverted_dict:
                    inverted_dict[value].append(key)
                else:
                    inverted_dict[value] = key
        return inverted_dict

    def get_cost(self, day, from_airport, to_airport):
        # Retrieve flights for the specified day and day 0
        flights_day = self.flights_by_day_dict.get(day, [])
        flights_day_0 = self.flights_by_day_dict.get(0, [])

        # Find the cost for the specified day
        cost_day = next(
            (
                flight[2]
                for flight in flights_day
                if flight[0] == from_airport and flight[1] == to_airport
            ),
            float("inf"),
        )

        # Find the cost for day 0
        cost_day_0 = next(
            (
                flight[2]
                for flight in flights_day_0
                if flight[0] == from_airport and flight[1] == to_airport
            ),
            float("inf"),
        )

        # Return the minimum cost if either exists, otherwise inf
        if cost_day == float("inf") and cost_day_0 == float("inf"):
            return float("inf")

        return min(cost_day, cost_day_0)

    def possible_flights_from_zone_to_zone_specific_day(self):
        areas_connections_by_day = {}

        for day, flights in self.flights_by_day_dict.items():
            areas_connections_list = []

            for flight in flights:
                connection = f"{self.area_by_airport.get(flight[0])} to {self.area_by_airport.get(flight[1])}"
                if connection not in areas_connections_list:
                    areas_connections_list.append(connection)

            areas_connections_by_day[day] = areas_connections_list

        return areas_connections_by_day

    def get_airports_by_areas(self):
        area_num = int(self.info[0][0])
        return {f"{i}": self.info[2 + i * 2] for i in range(0, area_num)}

    def get_list_of_airports(self):
        unique_airports = set()

        # Iterate through each sublist and add elements to the set
        for sublist in self.airports_by_area.values():
            for airport in sublist:
                unique_airports.add(airport)

        return list(unique_airports)

    def associated_area_to_airport(self, airport):
        return next(
            (
                area
                for area, airports in self.airports_by_area.items()
                if airport in airports
            ),
            "Airport not found",
        )

    def remove_duplicate(self, flights_by_day):
        for day, flights in flights_by_day.items():
            unique_flights = {}
            for flight in flights:
                flight_key = (flight[0], flight[1])
                if flight_key not in unique_flights:
                    unique_flights[flight_key] = flight
                else:
                    if flight[2] < unique_flights[flight_key][2]:
                        # print(flight[0],flight[1],flight[2],flight_key,unique_flights[flight_key][2])
                        unique_flights[flight_key] = flight
                flights_by_day[day] = list(unique_flights.values())
        return flights_by_day

    def possible_flights_from_an_airport_at_a_specific_day(self, day, from_airport):
        daily_flights = self.flights_by_day_dict.get(day, [])

        flights_from_airport = []
        for flight in daily_flights:
            if flight[0] == from_airport:

                flights_from_airport.append([flight[1], flight[2]])

        return flights_from_airport

    def possible_flights_from_an_airport_at_a_specific_day_with_previous_areas(
        self, day, from_airport, visited_areas
    ):
        daily_flights = self.flights_by_day_dict.get(
            day, []
        ) + self.flights_by_day_dict.get(0, [])
        flights_from_airport = []
        for flight in daily_flights:
            # print(self.associated_area_to_airport(airport=flight[0]))
            if (flight[0] == from_airport) and (
                self.associated_area_to_airport(airport=flight[1]) not in visited_areas
            ):

                flights_from_airport.append([flight[1], flight[2]])

        return flights_from_airport

    def which_area_to_explore(self, airports_by_area):
        return list(
            {
                key: len(value)
                for key, value in airports_by_area.items()
                if len(value) > 1
            }
        )
