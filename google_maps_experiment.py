import pandas as pd
import numpy as np
import pickle
import googlemaps
import re
import geopy.distance as distance
import params
import os.path

from utils import * 
gmaps = googlemaps.Client(key=params.gmapskey)
"""The keys in manual map are the top 100 keys for which X,Y is missing for locations. I was investigating locations on the map that did not have X,Y values. I wrote out a few, but realized that most of them were probably removed intentionally."""
manual_map = {
  'BATTERY ST TUNNEL NB BETWEEN ALASKAN WY VI NB AND AURORA AVE N':                 (np.nan, np.nan),
  'BATTERY ST TUNNEL SB BETWEEN AURORA AVE N AND ALASKAN WY VI SB':                 (np.nan, np.nan),
  'ALASKAN WY VI NB BETWEEN S ROYAL BROUGHAM WAY ON RP AND SENECA ST OFF RP':       (47.597705, -122.335437),
  'ALASKAN WY VI SB BETWEEN COLUMBIA ST ON RP AND ALASKAN WY VI SB EFR OFF RP':     (47.602001, -122.336429),
  'ALASKAN WY VI NB BETWEEN SENECA ST OFF RP AND WESTERN AV OFF RP':                (47.605535, -122.339018),
  'ALASKAN WY VI SB BETWEEN ELLIOTT AV ON RP AND COLUMBIA ST ON RP':                (47.602634, -122.336493),
  '1ST AVE S BETWEEN 1ST AVS ON N RP AND S ROYAL BROUGHAM WAY':                     (47.592461, -122.334868),
  'AURORA AVE N BETWEEN BATTERY ST TUNNEL NB AND THOMAS ST':                        (np.nan, np.nan),
  'RAINIER AVE S BETWEEN CORNELL AVE S AND 75TH AVE S':                             (47.511014700000004, -122.2438365),
  'BATTERY ST TUN ON RP BETWEEN BELL ST AND ALASKAN WY VI NB':                      (np.nan, np.nan),
  'RAINIER AVE S BETWEEN S STEVENS ST AND M L KING JR WAY S':                       (47.576671, -122.296800),
  'ALASKAN WY VI NB BETWEEN BATTERY ST TUN ON RP AND WESTERN AV OFF RP':            (np.nan, np.nan),
  'WESTERN AV OFF RP BETWEEN ALASKAN WY VI NB AND BELL ST':                         (np.nan, np.nan),
  'ALASKAN WY VI SB BETWEEN BATTERY ST TUNNEL SB AND ELLIOTT AV ON RP':             (np.nan, np.nan),
  'ALASKAN WAY S AND S KING ST':                                                    (47.598231, -122.335774),
  'DEXTER AVE N BETWEEN WARD ST AND PROSPECT ST':                                   (np.nan, np.nan),
  '35TH AVE SW AND FAUNTLEROY N WAY SW':                                            (np.nan, np.nan),
  'ALASKAN WAY S BETWEEN S ROYAL BROUGHAM WAY AND S ATLANTIC ST':                   (47.591872, -122.337013),
  'BROAD ST BETWEEN 5TH AVE N AND TAYLOR AVE N':                                    (np.nan, np.nan),
  'ELLIOTT AV ON RP BETWEEN ALASKAN WY VI SB AND ELLIOTT AVE':                      (np.nan, np.nan),
  'TERRY AVE BETWEEN SPRUCE ST AND ALDER ST':                                       (np.nan, np.nan),
  'ALASKAN WAY S BETWEEN ALASKAN E RDWY WAY S AND S ROYAL BROUGHAM WAY':            (np.nan, np.nan),
  'SENECA ST OFF RP BETWEEN ALASKAN WY VI NB AND 1ST AVE':                          (47.605496, -122.339078),
  'ALASKAN WAY S BETWEEN S KING ST AND ALASKAN E RDWY WAY S':                       (np.nan, np.nan),
  'ALASKAN WY VI NB BETWEEN BATTERY ST TUNNEL NB AND BATTERY ST TUN ON RP':         (np.nan, np.nan),
  'SPRUCE ST BETWEEN 8TH AVE AND TERRY AVE':                                        (np.nan, np.nan),
  'WEST SEATTLE BR WB BETWEEN 4TH AV S ON RP AND 1ST AV S ON RP':                   (np.nan, np.nan),
  'COLUMBIA ST ON RP BETWEEN 1ST AVE AND ALASKAN WY VI SB':                         (np.nan, np.nan),
  'BROAD ST BETWEEN BROAD ST EB OFF RP AND 9TH AVE N':                              (np.nan, np.nan),
  'ALASKAN WY VI NB AND BATTERY ST TUN ON RP':                                      (np.nan, np.nan),
  'WEST SEATTLE BR WB BETWEEN 4TH AV S OFF RP AND 4TH AV S ON RP':                  (np.nan, np.nan),
  'SR509 NB BETWEEN HOLDEN ST ON RP AND 1ST AV S BR NB':                            (np.nan, np.nan),
  'BATTERY ST TUN OFF RP BETWEEN ALASKAN WY VI SB AND WESTERN AVE':                 (np.nan, np.nan),
  'DENNY WAY BETWEEN PONTIUS AVE N AND YALE AVE':                                   (np.nan, np.nan),
  'ALASKAN WY VI SB AND COLUMBIA ST ON RP':                                         (np.nan, np.nan),
  'GOLF DR S BETWEEN 12TH AVE S AND S CHARLES ST':                                  (47.594293, -122.316573),
  'MERCER ST BETWEEN BROAD ST EB OFF RP AND 9TH AVE N':                             (np.nan, np.nan),
  'PONTIUS AVE N BETWEEN DENNY WAY AND JOHN W ST':                                  (np.nan, np.nan),
  'DENNY WAY AND PONTIUS AVE N':                                                    (np.nan, np.nan),
  'MERCER ST OFF RP BETWEEN FAIRVIEW AVE N AND FAIRVIEW AV OFF RP':                 (47.624673, -122.334335),
  'FAIRVIEW AVE N AND FAIRVIEW AV OFF RP':                                          (np.nan, np.nan),
  'BROAD ST BETWEEN HARRISON ST AND BROAD ST EB ON RP':                             (np.nan, np.nan),
  '4TH AVE S AND XW WELLER':                                                        (np.nan, np.nan),
  'FAIRVIEW AV OFF RP BETWEEN FAIRVIEW AVE N AND MERCER ST OFF RP':                 (47.624673, -122.334335),
  'SW ADMIRAL WAY BETWEEN 36TH AVE SW AND 37TH E AVE SW':                           (np.nan, np.nan),
  'AURORA AV SB OFF RP BETWEEN HARRISON ST AND AURORA AVE N':                       (47.622089, -122.343786),
  'S SPOKANE SR ST BETWEEN DUWAMISH AVE S AND EAST MARGINAL WAY S':                 (np.nan, np.nan),
  '1ST AV S OFF RP BETWEEN ALASKAN WY VI SB AND S ROYAL BROUGHAM WAY':              (np.nan, np.nan),
  '1ST AV S ON RP BETWEEN ALASKAN WY VI NB AND 1ST N AVE S':                        (np.nan, np.nan),
  'ALASKAN WY VI SB EFR OFF RP BETWEEN ALASKAN WY VI SB AND S ATLANTIC ST':         (np.nan, np.nan),
  'BROAD ST BETWEEN TAYLOR AVE N AND HARRISON ST':                                  (np.nan, np.nan),
  'S HOLLY PARK DR BETWEEN 38TH AVE S AND S LYON CT':                               (np.nan, np.nan),
  '2ND AVE N AND THOMAS ST':                                                        (np.nan, np.nan),
  'ALASKAN WAY S AND S ROYAL BROUGHAM WAY':                                         (np.nan, np.nan),
  'FAIRVIEW AVE N BETWEEN FAIRVIEW AV OFF RP AND ROY ST':                           (47.624974, -122.334368),
  'AURORA AVE N AND BATTERY ST TUNNEL NB':                                          (np.nan, np.nan),
  'S ROYAL BROUGHAM WAY BETWEEN ALASKAN WAY S AND ALASKAN E RDWY WAY S':            (np.nan, np.nan),
  '8TH AVE S BETWEEN YESLER WAY AND DEAD END 1':                                    (np.nan, np.nan),
  '7TH SB AVE N AND JOHN ST':                                                       (np.nan, np.nan),
  'GOLDEN GARDENS DR NW AND SEAVIEW PL NW':                                         (np.nan, np.nan),
  'AURORA AV NB OFF RP BETWEEN AURORA AVE N AND MERCER ST':                         (47.624423, -122.344029),
  'S ROYAL BROUGHAM WAY ON RP BETWEEN S ROYAL BROUGHAM WAY AND ALASKAN WY VI NB':   (np.nan, np.nan),
  'ALASKAN E RDWY WAY S BETWEEN S DEARBORN ST AND S ROYAL BROUGHAM WAY':            (np.nan, np.nan),
  'SW KLICKITAT NR WAY AND SW SPOKANE NR ST':                                       (np.nan, np.nan),
  'ALASKAN WAY S AND S DEARBORN ST':                                                (np.nan, np.nan),
  'BROAD ST BETWEEN BROAD ST EB ON RP AND BROAD ST WB ON RP':                       (np.nan, np.nan),
  'S ROYAL BROUGHAM WAY BETWEEN ALASKAN E RDWY WAY S AND EAST FRONTAGE RD S':       (np.nan, np.nan),
  'S ALASKA ST BETWEEN 30TH AVE S AND M L KING JR WR WAY S':                        (np.nan, np.nan),
  'S SPOKANE NR ST BETWEEN 1ST AV S ON RP AND 1ST AVE S':                           (47.571437, -122.334114),
  '4TH AVE S AND 4TH AV S OFF RP':                                                  (np.nan, np.nan),
  'ALASKAN WY VI NB AND S ROYAL BROUGHAM WAY ON RP':                                (np.nan, np.nan),
  'ALASKAN WY VI NB AND SENECA ST OFF RP':                                          (47.605496, -122.339078),
  '4TH AV S OFF RP AND WEST SEATTLE BR WB':                                         (np.nan, np.nan),
  '6TH AVE S BETWEEN S ATLANTIC ST AND S MASSACHUSETTS ST':                         (np.nan, np.nan),
  '4TH AV S OFF RP BETWEEN 4TH AVE S AND WEST SEATTLE BR WB':                       (np.nan, np.nan),
  'RAINIER AVE S AND S STEVENS ST':                                                 (47.576884, -122.297074),
  'BROAD ST EB OFF RP BETWEEN BROAD ST AND MERCER SR ST':                           (np.nan, np.nan),
  'BROAD ST BETWEEN DEXTER AVE N AND ROY W ST':                                     (np.nan, np.nan),
  'DUWAMISH AVE S AND S SPOKANE SR ST':                                             (np.nan, np.nan),
  'BROADWAY AND SPRUCE ST':                                                         (np.nan, np.nan),
  'ALASKAN WY VI SB AND ALASKAN WY VI SB EFR OFF RP':                               (np.nan, np.nan),
  'BROAD ST AND TAYLOR AVE N':                                                      (np.nan, np.nan),
  'ALASKAN WY VI SB AND ELLIOTT AV ON RP':                                          (np.nan, np.nan),
  'ALASKAN WAY AND ALASKAN E RDWY WAY':                                             (np.nan, np.nan),
  '15TH AVE NE BETWEEN NE BOAT E ST AND NE COLUMBIA RD':                            (np.nan, np.nan),
  'FAIRVIEW AV OFF RP AND MERCER ST OFF RP':                                        (np.nan, np.nan),
  'MONTLAKE BLVD E AND MONTLAKE BV EB ON RP':                                       (np.nan, np.nan),
  'STONE AVE N AND N 135TH ST':                                                     (np.nan, np.nan),
  '17TH AVE SW BETWEEN SW CAMBRIDGE ST AND DELRIDGE WAY SW':                        (np.nan, np.nan),
  'LAKE WASHINGTON BLVD S AND S JUNEAU ST':                                         (np.nan, np.nan),
  'S HINDS ST BETWEEN HINDS N PL S AND HINDS S PL S':                               (np.nan, np.nan),
  'ALASKAN WY VI NB AND WESTERN AV OFF RP':                                         (np.nan, np.nan),
  '4TH AVE S BETWEEN 4TH AV S OFF RP AND S SPOKANE NR ST':                          (np.nan, np.nan),
  '32ND AVE S AND M L KING JR ER WAY S':                                            (np.nan, np.nan),
  'BROADWAY BETWEEN SPRUCE ST AND E SPRUCE ST':                                     (np.nan, np.nan),
  'ALASKAN WY VI NB AND BATTERY ST TUNNEL NB':                                      (np.nan, np.nan),
  'CORGIAT DR S AND S CORGIAT DR':                                                  (np.nan, np.nan),
  'S SPOKANE NR ST BETWEEN 4TH AV S ON RP AND 4TH AVE S':                           (np.nan, np.nan),
  'I5 SB AND 145TH ST ON RP':                                                       (np.nan, np.nan),
  'BROADWAY BETWEEN TERRY AVE AND SPRUCE ST':                                       (np.nan, np.nan),
  'ALASKAN WY VI SB AND BATTERY ST TUNNEL SB':                                      (np.nan, np.nan),
  '28TH AVE SW BETWEEN SW BRANDON ST AND DEAD END':                                 (np.nan, np.nan),
  '40TH AVE NE BETWEEN NE 46TH ST AND SAND POINT WAY NE':                           (47.662323, -122.284793),
  'SYLVAN WAY SW BETWEEN HIGH POINT DR SW AND SW HOLLY ST':                         (np.nan, np.nan),
  'BONAIR PL SW BETWEEN 53RD AVE SW AND DEAD END W':                                (np.nan, np.nan),
  '1ST AVE S AND 1ST AVS ON N RP':                                                  (np.nan, np.nan),
  'HARRISON ST BETWEEN BROAD ST AND TAYLOR AVE N':                                  (np.nan, np.nan),
  'ALASKAN WAY S AND XW WASHINGTON':                                                (np.nan, np.nan),
  'UNIVERSITY BR OFF RP AND NE 40TH ST':                                            (np.nan, np.nan),
  '46TH AVE SW AND SW 100TH ST':                                                    (np.nan, np.nan),
  '41ST AVE S AND S ALASKA ST':                                                     (np.nan, np.nan),
  'TAYLOR AVE N BETWEEN BROAD ST AND HARRISON ST':                                  (np.nan, np.nan),
  '6TH AVE N AND RAYE LOWER ST':                                                    (np.nan, np.nan),
  '15TH AVE E AND E BOSTON ST':                                                     (np.nan, np.nan),
  '68TH AVE S AND HOLYOKE WAY S':                                                   (np.nan, np.nan),
  'S COLUMBIAN EB WAY BETWEEN WEST SEATTLE BR EB AND I5 NB - COLUMBIAN WY RP':      (np.nan, np.nan),
  'NW 100TH PL BETWEEN NW 100TH ST AND 8TH AVE NW':                                 (np.nan, np.nan),
  '5TH PL S BETWEEN KENYON ON RP AND 5TH AVE S':                                    (np.nan, np.nan),
  'DUWAMISH RIVER TRL AND WEST MARGINAL WAY SW':                                    (np.nan, np.nan),
  '27TH AVE S AND S DELAPPE PL':                                                    (np.nan, np.nan),
  'SAND POINT WAY NE BETWEEN 41ST S AVE NE AND 41ST N AVE NE':                      (np.nan, np.nan),
  'REPUBLICAN ST BETWEEN 6TH AVE N AND AURORA AVE N':                               (np.nan, np.nan),
  '2ND AVE S BETWEEN DEAD END N AND S KENYON ST':                                   (np.nan, np.nan),
  '30 UPPER AVE W AND W DRAVUS ST':                                                 (np.nan, np.nan),
  'BONAIR PL SW BETWEEN DEAD END E AND 53RD AVE SW':                                (np.nan, np.nan),
  'W MERCER ST BETWEEN DEAD END AND ELLIOTT AVE W':                                 (np.nan, np.nan),
  'HOLDEN ST ON RP AND SR509 NB':                                                   (47.534211, -122.333102),
  'BROADWAY AND TERRY AVE':                                                         (np.nan, np.nan),
  '1ST AVE S AND RAILROAD CR WAY S':                                                (np.nan, np.nan),
  'TERRY AVE BETWEEN BROADWAY AND SPRUCE ST':                                       (np.nan, np.nan),
  '16TH AVE W AND W GALER ST':                                                      (np.nan, np.nan),
  '39TH AVE S BETWEEN BEACON ER S AVE S AND S BARTON ST':                           (np.nan, np.nan),
  'WEST SEATTLE BRIDGE TRL BETWEEN SW SPOKANE ST AND WEST SEATTLE BR NR TRL':       (np.nan, np.nan),
  'S HANFORD ST BETWEEN HINDS PL S AND 15TH AVE S':                                 (np.nan, np.nan),
  '31ST AVE S AND M L KING JR ER WAY S':                                            (np.nan, np.nan),
  'CARR PL N BETWEEN N 34TH ST AND N 35TH ST':                                      (np.nan, np.nan),
  'JOHN ST BETWEEN PONTIUS W AVE N AND PONTIUS E AVE N':                            (np.nan, np.nan),
  'WESTLAKE EAST RDWY AVE N BETWEEN WESTLAKE AVE N AND WESTLAKE SHORE RDWY AVE N':  (np.nan, np.nan),
  'GREENWOOD AVE N AND XW N85-N87':                                                 (np.nan, np.nan),
  'EAST FRONTAGE RD S BETWEEN ALASKAN WY VI SB EFR OFF RP AND S ATLANTIC ST':       (np.nan, np.nan),
  'ALASKAN WAY W BETWEEN W LEE ST AND W GALER ST':                                  (np.nan, np.nan),
  '23RD AVE E AND E LYNN ST':                                                       (np.nan, np.nan),
  'ALLEY BETWEEN SOUTH AND UNKNOWN':                                                (np.nan, np.nan),
  'ALASKAN WAY W AND W GALER ST':                                                   (np.nan, np.nan),
  'ELLIOTT BAY TRL AND PIER 91 ACCESS RD':                                          (np.nan, np.nan),
  '2ND AVE NE AND DEAD END 6':                                                      (np.nan, np.nan),
  'ALASKAN E RDWY WAY S BETWEEN ALASKAN WAY S AND S DEARBORN ST':                   (np.nan, np.nan),
  '11TH AVE NE BETWEEN NE 135TH S ST AND NE 135TH N ST':                            (np.nan, np.nan),
  'SW ORCHARD ST BETWEEN PARSHALL PL SW AND CALIFORNIA AVE SW':                     (np.nan, np.nan),
  'S DEARBORN ST BETWEEN RAILROAD CR WAY S AND 1ST AVE S':                          (np.nan, np.nan),
  'E HIGHLAND DR AND DEAD END 2':                                                   (np.nan, np.nan),
  'NE 46TH ST BETWEEN 40TH AVE NE AND 41ST AVE NE':                                 (np.nan, np.nan),
  'WESTLAKE SHORE RDWY AVE N BETWEEN WESTLAKE EAST RDWY N AVE N AND HIGHLAND DR':   (np.nan, np.nan),
  '32ND AVE SW BETWEEN SW GRAHAM ST AND SW MORGAN ST':                              (np.nan, np.nan),
  'W EMERSON ST AND W ROBERTS WAY':                                                 (np.nan, np.nan),
  'S HANFORD ST BETWEEN 6TH AVE S AND 8TH AVE S':                                   (np.nan, np.nan),
  'S COLUMBIAN WB WAY BETWEEN WEST SEATTLE BR WB AND COLUMBIAN WY - I5 SB RP':      (np.nan, np.nan),
  '9TH AVE S AND S ADAMS ST':                                                       (np.nan, np.nan),
  'NE 125TH ST AND XW 26NE-27NE':                                                   (np.nan, np.nan),
  'CENTRAL AND UNKNOWN':                                                            (np.nan, np.nan),
  'RENTON AVE S AND S OREGON W ST':                                                 (np.nan, np.nan),
  'AIRPORT WAY S AND CITY LIMITS N':                                                (np.nan, np.nan),
  'ALLEY BETWEEN CENTRAL AND UNKNOWN':                                              (np.nan, np.nan),
  '17TH AVE NE AND NE 123RD ST':                                                    (np.nan, np.nan),
  'SYLVAN LN SW AND DEAD END':                                                      (np.nan, np.nan),
  '51ST PL SW AND SW ALASKA ST':                                                    (np.nan, np.nan),
  'NE BOAT ST AND NE COLUMBIA RD':                                                  (np.nan, np.nan),
  '1ST PL NE AND NE 117TH ST':                                                      (np.nan, np.nan),
  '38TH AVE S BETWEEN S EDDY ST AND M L KING JR ER WAY S':                          (np.nan, np.nan),
  '42ND AVE S AND DEAD END 1':                                                      (np.nan, np.nan),
  'S ANDOVER ST AND DEAD END 5':                                                    (np.nan, np.nan),
  '16TH AVE SW BETWEEN DEAD END AND SW FLORIDA ST':                                 (np.nan, np.nan),
  '25TH AVE NE BETWEEN NE 126TH ST AND NE 127TH ST':                                (np.nan, np.nan),
  'CHELAN AVE SW AND DEAD END':                                                     (np.nan, np.nan),
  'RAVENNA AVE NE BETWEEN RAVENNA PL NE AND NE 55TH N ST':                          (np.nan, np.nan),
  '3RD AVE AND XW JEFFERSON':                                                       (np.nan, np.nan),
  'SPRUCE ST AND TERRY AVE':                                                        (np.nan, np.nan),
  '8TH AVE NW AND NW 100TH PL':                                                     (np.nan, np.nan),
  '22ND WR AVE NE AND NE 47TH ST':                                                  (np.nan, np.nan),
  'N 135TH ST BETWEEN DEAD END AND LINDEN AVE N':                                   (np.nan, np.nan),
  'LAKEVIEW LN NE AND NE 105TH ST':                                                 (np.nan, np.nan),
  'GALER ST AND TAYLOR UPPER AVE N':                                                (np.nan, np.nan),
  'SPRUCE ST BETWEEN TERRY AVE AND BROADWAY':                                       (np.nan, np.nan),
  '31ST AVE SW AND SW CAMBRIDGE ST':                                                (np.nan, np.nan),
  'E JAMES WAY AND XW 10TH AVE':                                                    (np.nan, np.nan),
  'WAVERLY PL N AND WHEELER ST':                                                    (np.nan, np.nan),
  '35TH AVE SW BETWEEN SW LANHAM WAY AND SW GRAHAM ST':                             (np.nan, np.nan),
  'YUKON AVE S BETWEEN SPEAR PL S AND DEAD END':                                    (np.nan, np.nan),
  '69TH PL S AND S 116TH PL':                                                       (np.nan, np.nan),
  '3RD AVE S AND S KING ST':                                                        (np.nan, np.nan),
  '23RD AVE NE AND NE 83RD ST':                                                     (np.nan, np.nan),
  '11TH CR AVE SW BETWEEN SW SPOKANE ST AND WEST SEATTLE BRIDGE TRL':               (np.nan, np.nan),
  'RAILROAD WR WAY S BETWEEN S KING ST AND RAILROAD CR WAY S':                      (np.nan, np.nan),
  'SYLVAN WAY SW AND SW SYLVAN HEIGHTS DR':                                         (np.nan, np.nan),
  'NW CANAL ST BETWEEN 1ST AVE NW AND NW CANAL ST':                                 (np.nan, np.nan),
  '47TH AVE SW AND DEAD END 6':                                                     (np.nan, np.nan),
  '30TH AVE S AND S FRONTENAC ST':                                                  (np.nan, np.nan),
  'S 114TH ST AND CITY LIMITS':                                                     (np.nan, np.nan),
  '2ND AVE S BETWEEN S FIDALGO ST AND DEAD END 2':                                  (np.nan, np.nan),
  'E YESLER WAY AND DEAD END 1':                                                    (np.nan, np.nan),
  'RAVENNA AVE NE AND RAVENNA PL NE':                                               (np.nan, np.nan),
  'E LYNN ST BETWEEN 22ND AVE E AND 23RD AVE E':                                    (np.nan, np.nan),
  '5TH AVE AND XW PIKE-UNION':                                                      (np.nan, np.nan),
  '27TH AVE NE AND NE 113TH ST':                                                    (np.nan, np.nan),
  '31ST AVE SW AND SW HINDS ST':                                                    (np.nan, np.nan),
  'BURKE GILMAN TRL AND NE 77TH ST':                                                (np.nan, np.nan),
  'M L KING JR ER WAY S BETWEEN 31ST AVE S AND S ANGELINE ST':                      (np.nan, np.nan),
  'SOUTH AND UNKNOWN':                                                              (np.nan, np.nan),
  'W GARFIELD ST AND DEAD END 1':                                                   (np.nan, np.nan),
  'ALOHA ST BETWEEN WESTLAKE N AVE N AND WESTLAKE EAST RDWY AVE N':                 (np.nan, np.nan),
  '30TH AVE SW AND DEAD END 1':                                                     (np.nan, np.nan),
  'N NORTHGATE WAY AND XW STONE-INTERLAKE':                                         (np.nan, np.nan),
  '12TH AVE S AND GOLF DR S':                                                       (np.nan, np.nan),
  '25TH AVE S AND S HANFORD E ST':                                                  (np.nan, np.nan),
  '15TH AVE SW AND SW MYRTLE ST':                                                   (np.nan, np.nan),
  '58TH AVE NE BETWEEN NE 69TH ST AND NE 70TH ST':                                  (np.nan, np.nan),
  'ALOHA ST AND WESTLAKE EAST RDWY AVE N':                                          (np.nan, np.nan),
  '58TH AVE SW AND SW STEVENS ST':                                                  (np.nan, np.nan),
  'NE 125TH ST BETWEEN SAND POINT ER WAY NE AND DEAD END 2':                        (np.nan, np.nan),
  'WESTLAKE AVE N BETWEEN WESTLAKE EAST RDWY AVE N AND 8TH S AVE N':                (np.nan, np.nan),
  '6TH AVE S AND S ATLANTIC ST':                                                    (np.nan, np.nan), }


# try to fill the locations that have a specific format
def try_between_and_match(sa):
  m = re.match('(.*) BETWEEN (.*) AND (.*)', sa)
  if m is not None:
    
    i1 = m.group(1)+' & '+m.group(2)
    i2 = m.group(1)+' & '+m.group(3)
    
    geo1 = gmaps.geocode(i1)
    geo2 = gmaps.geocode(i2)
    if len(geo1) == 1 and len(geo2) == 1:
      l1 =  (geo1[0]['geometry']['location']['lat'], geo1[0]['geometry']['location']['lng'])
      l2 =  (geo2[0]['geometry']['location']['lat'], geo2[0]['geometry']['location']['lng'])
      print(distance.distance(l1, l2).mi)
      if distance.distance(l1, l2).mi < 1:
          lng = (geo1[0]['geometry']['location']['lng'] + geo2[0]['geometry']['location']['lng'])/2
          lat = (geo1[0]['geometry']['location']['lat'] + geo2[0]['geometry']['location']['lat'])/2
          print(lat, lng)
          return lat, lng
  return None, None
  
def try_fix_with_gmaps():
  '''This was an attempt to see what google maps has to offer toward filling the X,Y values for rows with LOCATION but no XY. I decided X,Y were intentionally removed so this is not needed.'''
  pd.set_option('display.max_rows', None)  
  pd.set_option('display.max_colwidth', None)  
  pd.set_option('display.max_columns', None)  
  df = open_collisions()

  # create dataframe indexed by incident tatetime
  df2 = df.set_index('INCDTTM')
  
  print('incidents by month')
  print(df2.groupby(pd.Grouper(freq='Q'))['COLDETKEY'].count())
  print('missing X values by month')
  print(df2[df2['X'].isna()].groupby(pd.Grouper(freq='Q'))['COLDETKEY'].count())
  
  df3 = df2[df2.index >= '2020-01-01']
  df3[df3['X'].isna()]['LOCATION']
  df3[df3['X'].isna()]['LOCATION'].value_counts(dropna = False)
  location_to_yx = dict()

  

  # build map of location to x y, also check if there are locations that map to two different yx pairs.
  for i, (location, (y, x)) in enumerate(zip(df['LOCATION'], zip(df['Y'], df['X']))):
    if pd.isna(x):
      continue
    if location in location_to_yx:
      if location_to_yx[location] != (y, x):
        print('found discrepancy', location, location_to_yx[location], y, x, 'dist', distance.distance((y, x), location_to_yx[location]).mi)
    else:
      location_to_yx[location] = (y, x)

  
  loc_count = df['LOCATION'].value_counts(dropna=False)
  print('top locations')
  print(loc_count[:20])
  loc_count_df = loc_count.to_frame('counts')
  loc_count_df.insert(1, 'YX', [location_to_yx[sa] if sa in location_to_yx else np.nan for sa in loc_count.index])
  print('top missing locations')
  print(loc_count_df[loc_count_df['YX'].isna()][:20]['counts'])
  
  # step through individual locations with no y,x that have 'between..and..' format and see if we can fix them
  # spoiler, in general we cannot. this doesn't work very well.
  for i in df[~df['LOCATION'].isna() & df['X'].isna()].index:
    print(i)
    assert pd.isna(df.iloc[i]['Y'])
    l = df.iloc[i]['LOCATION']
    lat, lng = try_between_and_match(l)
    if lat is not None:
      print('updating', l)
      df.iloc[i]['X'] = lng
      df.iloc[i]['Y'] = lat
      input()
      
      
if __name__ == '__main__':
    try_fix_with_gmaps()