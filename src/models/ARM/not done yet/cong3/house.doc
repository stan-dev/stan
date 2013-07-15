
Documentation for U.S. House of Representatives Electoral Data

Send comments and questions:  Gary King, Department of Government,
Harvard University, Littauer Center North Yard, Cambridge, Massachusetts
02138; Email: gk@isr.Harvard.Edu.  Thanks goes to Michael Ting for managing
the latest and most extensive round of data verification, and to the
National Science Foundation for grant SBR-9223637.


---------
CITATIONS
---------
Earlier versions of these data were used and gradually improved in the
context of research reported in the following publications, among others.

Gelman, Andrew and Gary King. "A Unified Method of Evaluating Electoral
Systems and Redistricting Plans," American Journal of Political Science,
forthcoming, Vol. 38, No. 2 (May, 1994).

Gelman, Andrew and Gary King. "Estimating Incumbency Advantage Without
Bias," American Journal of Political Science, Vol. 34, No. 4 (November,
1990): Pp. 1142--1164.

Gelman, Andrew and Gary King. "Estimating the Electoral Consequences of
Legislative Redistricting," Journal of the American Statistical
Association, Vol. 85, No. 410 (June, 1990): Pp. 274--282, (with Andrew
Gelman).

King, Gary. "Constituency Service and Incumbency Advantage," British
Journal of Political Science, Vol. 21, No. 1 (January, 1991): Pp. 119--128.

King, Gary. "Stochastic Variation: A Comment on Lewis-Beck and Skalaban's
`The R-Square'," Political Analysis, Vol. 2 (1991): Pp. 185--200.

King, Gary. "Electoral Responsiveness and Partisan Bias in Multiparty
Democracies," Legislative Studies Quarterly, Vol. XV, No. 2 (May, 1990):
Pp. 159--181.

King, Gary and Andrew Gelman. "Systemic Consequences of Incumbency
Advantage in the U.S. House," American Journal of Political Science, Vol.
35, No. 1 (February, 1991): Pp. 110--138.


--------
OVERVIEW
--------
These files contain data for every US House of Representatives election
from 1896 to 1992, inclusive.  The results from each election appear in a
separate file named with the election year and extension ".ASC" (e.g.,
"1934.ASC" is the 1934 House election).

For unusual elections (such as minor party winners, or two Democrats and no
Republican candidates, etc.), we have constructed an exceptions file,
EXCEPTH.ASC.  Each election appears in either ****.ASC or EXCEPTH.ASC, but
not both.  See details below.


----------------------------
"****.ASC" FILE ORGANIZATION
----------------------------
Each file is formatted identically with blanks between columns and with the
following fields: (1) State #, (2) District #, (3) Incumbency Code, (4) #
Democratic votes, and (5) # Republican votes.  In all cases, a value of -9
was entered when there was either no data available or appropriate for that
field.  Vote totals can always be compared between elections with the same
state and district number between major redistricting periods.  For
example, the election files corresponding to 1942-1950 will all have the
same number of rows.  Redistrictings that do not occur immediately prior to
years ending in "2" are dealt with by using new district numbers.  The
fields are as follows:

State number:  States are numbered using the standard ICPSR code:

     New England            Middle Atlantic
        01  Connecticut        11  Delaware
        02  Maine              12  New Jersey
        03  Massachusetts      13  New York
        04  New Hampshire      14  Pennsylvania
        05  Rhode Island
        06  Vermont

     East North Central     West North Central
        21  Illinois           31  Iowa
        22  Indiana            32  Kansas
        23  Michigan           33  Minnesota
        24  Ohio               34  Missouri
        25  Wisconsin          35  Nebraska
                               36  N. Dakota
                               37  S. Dakota

     Solid South
        40  Virginia           45  Louisiana
        41  Alabama            46  Mississippi
        42  Arkansas           47  N. Carolina
        43  Florida            48  S. Carolina
        44  Georgia            49  Texas

     Border States          Mountain States
        51  Kentucky           61  Arizona
        52  Maryland           62  Colorado
        53  Oklahoma           63  Idaho
        54  Tennessee          64  Montana
        55  Washington,D.C.    65  Nevada
        56  W. Virginia        66  New Mexico
                               67  Utah
                               68  Wyoming

     Pacific States          External States
        71  California          81  Alaska
        72  Oregon              82  Hawaii
        73  Washington

District #:  Districts are numbered as they are by law in each state, with
     the exception that At-Large districts are numbered starting from 98 in
     descending order.  Thus, a 10-district state which includes 2 at-large
     districts will have district numbers 1-8, 97, 98.  Each state will
     have the same number of districts in every file from any given
     redistricting period.  If the above state were to lose its at-large
     districts and change to 10 standard geographic districts before its
     next census-based redistricting, then each file from that
     redistricting period will have district numbers 1-10, 97, 98, with -9s
     filled in for nonexistent elections.
Incumbency Code:  0=Open seat, 1=Democratic incumbent, -1=Republican
     incumbent.
# Democratic votes:  The vote total for the Democratic candidate.  If the
     Democratic candidate was also endorsed by a third party, then we have
     added the two vote totals.
# Republican votes:  The vote total for the Republican candidate.  If the
     Republican candidate was also endorsed by a third party, then we have
     added the two vote totals.

     Who Won?  Winners can generally be determined from the files by seeing
     which candidate has more votes.  However, this only works for standard
     elections, where a Democrat runs against a Republican and one of them
     wins.  Occasionally, however, a 3d-party candidate will win, or the
     vote totals for the winner will not be available.  In order to
     preserve what was known about those elections, the following algorithm
     should be used to figure out winners:

     1. IF both parties have vote totals >= 0, THEN the party with more
          votes is the winner.
     2. IF both parties have vote totals = -9, THEN either no election took
          place, OR there is a 3d-Party winner, OR there was an at-large
          election.  In the latter two cases, the election results were
          recorded in the exceptions file (excepth.asc).
     3. IF one party has a vote total = 0 AND the other party has a vote
          total recorded as -9 then that election should be in the
          exceptions file (most of these are uncontested elections where
          the number of votes for the winner was not reported).
     4. IF one party has a vote total > 0 AND the other party has a vote
          total = -9, THEN the first party is the winner.


------------------------------------------
EXCEPTIONS:  EXCEPTH.ASC FILE ORGANIZATION
------------------------------------------
This file contains information on elections which are not describable as a
single Democrat running against a single Republican.  This file is in ASCII
format with spaces between the columns.  The fields follow:

year     = the year in four digits
state    = the ICPSR state number (as described above)
district = district number (as described above),
Dem      = 1 if the winning candidate is Democratic, and 0 otherwise
Rep      = 1 if the winning candidate is Republican, and 0 otherwise
Minor    = 1 if the winning candidate is neither a Democrat nor a Republican
Votes    = the number of votes received by the winning candidate, where
           available

SOURCES:

Election figures up to 1984 were taken from Congressional Quarterly's Guide
to US Elections (Second Edition, 1985).  According to p.1062 of this
volume, the vast majority of its data come from the ICPSR.  Non-ICPSR
sources are listed on that page.  However, many hundreds of errors still
existing in the ICPSR files were apparently corrected in CQ.  For some
information on these errors, see Andrew Gelman and Gary King. "Estimating
Incumbency Advantage Without Bias," American Journal of Political Science,
Vol. 34, No. 4 (November, 1990): Pp. 1142--1164.

Post-1984 data were taken from the official returns (as opposed to the
unofficial figures on election night) as listed by Congressional
Quarterly's Weekly Report.  These data typically come out around the March
or April after the election.  All files were checked extensively against
published lists of elected congressmen to make sure that the proper winners
could be produced with the data entered and the rules listed above.
