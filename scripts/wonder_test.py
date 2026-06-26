import urllib.request, urllib.parse
# Minimal CDC WONDER API test: Underlying Cause of Death 1999-2020 (D76)
# National deaths by year for infants (<1yr), cause = SIDS (ICD-10 R95)
params = [
 ("B_1","D76.V1-level1"),               # group by year
 ("M_1","D76.M1"),("M_2","D76.M2"),("M_3","D76.M3"),
 ("F_D76.V1","*All*"),
 ("F_D76.V5","1"),                       # age: ten-year group "1" = <1 yr
 ("V_D76.V5","1"),
 ("F_D76.V2","R95"),                     # ICD-10 underlying cause R95 (SIDS)
 ("O_V5_fmode","freg"),("O_age","D76.V5"),
 ("O_javascript","on"),("O_location","D76.V9"),
 ("O_precision","1"),("O_rate_per","100000"),
 ("O_show_totals","true"),("O_timeout","300"),
 ("O_title","wonder_test"),
 ("VM_D76.M6_D76.V1_S","*All*"),
]
xml = "<request-parameters>\n"
xml+= '<parameter><name>accept_datause_restrictions</name><value>true</value></parameter>\n'
for n,v in params:
    xml+=f"<parameter><name>{n}</name><value>{v}</value></parameter>\n"
xml+="</request-parameters>"
data = urllib.parse.urlencode({"request_xml":xml,"accept_datause_restrictions":"true"}).encode()
req=urllib.request.Request("https://wonder.cdc.gov/controller/datarequest/D76",data=data)
try:
    r=urllib.request.urlopen(req,timeout=60)
    body=r.read().decode(errors="replace")
    print("HTTP OK, len",len(body))
    print(body[:1500])
except urllib.error.HTTPError as e:
    print("HTTPError",e.code); print(e.read().decode(errors="replace")[:1500])
except Exception as e:
    print("ERR",type(e).__name__,e)
