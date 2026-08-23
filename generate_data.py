"""Built-in demo data generator for InventoryLoop.

Writes inventory_master.csv, variance_events.csv, branches.csv to ./data.
Models a REALISTIC ~15-branch mid-size HVAC/plumbing/electrical distributor
(the target ICP). Dollar magnitudes are calibrated to be believable:

  1. Count discrepancies land on CHEAP, FAST-moving SKUs (weight = velocity /
     unit_cost^0.6) — real errors cluster on cheap high-volume parts, not the
     $4,800 compressors nobody miscounts.
  2. Discrepancy size is capped at +/-1 unit for high-cost items (>$400).
  3. ~2.5 discrepancies / branch / day.

Run: python generate_data.py
"""
from __future__ import annotations
import json, random, uuid
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED)
DATA_DIR = Path(__file__).parent / "data"; DATA_DIR.mkdir(exist_ok=True)

N_SKUS = 4000
N_BRANCHES = 15
HISTORY_DAYS = 180
EVENTS_PER_BRANCH_DAY = 2.5
OPEN_RATIO = 0.10

CATEGORY_TEMPLATES = {
    "Refrigerant":     (["R-410A","R-32","R-454B","R-22","R-407C"], ["25lb cylinder","50lb jug","30lb tank","12oz can"], (95,850), "REF"),
    "Copper Tubing":   (["Soft Copper Tubing","Line Set","ACR Copper"], ['1/4 in. x 50ft','3/8 in. x 50ft','1/2 in. x 50ft','5/8 in. x 50ft'], (35,425), "CU"),
    "Compressor":      (["Scroll Compressor","Rotary Compressor","Variable Speed Compressor"], ["2 Ton 208V","2.5 Ton 230V","3 Ton 230V","4 Ton 460V","5 Ton 460V"], (450,4800), "COMP"),
    "Motor":           (["Condenser Fan Motor","Blower Motor","ECM Motor","PSC Motor"], ["1/6 HP","1/4 HP","1/3 HP","1/2 HP","3/4 HP","1 HP"], (85,685), "MOT"),
    "Controls":        (["Smart Thermostat","Furnace Control Board","Contactor","Capacitor","Relay"], ["24V","120V","30A","40A","45 MFD","70 MFD"], (12,385), "CTL"),
    "Pipe & Fittings": (["PVC Pipe","PEX Tubing","Copper Fitting Elbow","CPVC Pipe","Cast Iron Pipe"], ['1/2"','3/4"','1"','1-1/2"','2"','4"'], (3,180), "PIPE"),
    "Valves":          (["Ball Valve","Gate Valve","Check Valve","Pressure Reducing Valve"], ['1/2"','3/4"','1"','1-1/4"','2"'], (8,320), "VLV"),
    "Water Heaters":   (["Gas Water Heater","Electric Water Heater","Tankless Water Heater","Expansion Tank"], ["40 gal","50 gal","75 gal","199k BTU"], (280,2400), "WH"),
    "Electrical":      (["THHN Wire","Circuit Breaker","EMT Conduit","Disconnect Switch"], ["12 AWG","10 AWG","20A","30A","60A","3/4 in."], (6,240), "ELE"),
}
CAT_W = {"Refrigerant":.10,"Copper Tubing":.10,"Compressor":.06,"Motor":.08,"Controls":.12,"Pipe & Fittings":.18,"Valves":.10,"Water Heaters":.06,"Electrical":.20}
VEL_W = {"A":.20,"B":.30,"C":.50}
VEL_SCORE = {"A":1.0,"B":0.6,"C":0.3}
SOURCES = ["cycle_count","picking","stocking","branch_escape"]; SRC_W = [.40,.35,.15,.10]
STAGES = ["detected","investigating","physical_count","system_correction","resolved"]
HANDLERS = ["M. Hernandez","T. Wilson","S. Patel","J. Nguyen","R. Johnson","K. O'Brien","D. Rivera","L. Chen"]
CITIES = [("Columbia","SC"),("Charleston","SC"),("Greenville","SC"),("Spartanburg","SC"),("Myrtle Beach","SC"),
          ("Charlotte","NC"),("Raleigh","NC"),("Greensboro","NC"),("Asheville","NC"),("Wilmington","NC"),
          ("Augusta","GA"),("Atlanta","GA"),("Savannah","GA"),("Macon","GA")]

def _cost(cat):
    lo,hi = CATEGORY_TEMPLATES[cat][2]
    return round(float(np.clip(np.random.lognormal(np.log((lo+hi)/6),0.7)+lo, lo, hi)), 2)

def generate_inventory(today):
    cats = random.choices(list(CAT_W), weights=list(CAT_W.values()), k=N_SKUS)
    vels = random.choices(list(VEL_W), weights=list(VEL_W.values()), k=N_SKUS)
    rows=[]
    for i in range(N_SKUS):
        cat,vel = cats[i], vels[i]; names,sizes,_,pfx = CATEGORY_TEMPLATES[cat]
        nm=random.choice(names); sz=random.choice(sizes); tag=nm.split()[0].upper().replace("-","")[:5]
        rop=int(np.clip(np.random.normal({"A":50,"B":20,"C":8}[vel],8),2,200))
        days_back=int(np.clip(np.random.normal({"A":30,"B":75,"C":130}[vel]+45,30),5,320))
        rows.append({"sku_id":f"HVAC-{pfx}-{tag}-{i:05d}","description":f"{nm}, {sz}","category":cat,
            "unit_cost":_cost(cat),"velocity_class":vel,"reorder_point":rop,
            "reorder_qty":int(rop*np.random.uniform(1.5,3.0)),
            "last_cycle_count_date":(today-timedelta(days=days_back)).date().isoformat()})
    return pd.DataFrame(rows)

def generate_branches():
    b=[{"branch_id":"DC-01","branch_name":f"{CITIES[0][0]} Distribution Center","branch_type":"DC","city":CITIES[0][0],"state":CITIES[0][1]}]
    for i in range(1,N_BRANCHES):
        c=CITIES[i % len(CITIES)]
        b.append({"branch_id":f"BR-{100+i}","branch_name":f"{c[0]} Branch","branch_type":"branch","city":c[0],"state":c[1]})
    return pd.DataFrame(b)

def _stage_hist(det, hrs, is_open):
    sh=[0,0.45,0.30,0.20,0.05]; stop=random.choices([1,2,3,4],weights=[.4,.35,.2,.05])[0] if is_open else 5
    h=[]; cur=det; st="open"
    for idx,s in enumerate(STAGES):
        if idx>=stop: break
        d=hrs*sh[idx]
        if s=="detected": h.append({"stage":s,"started_at":cur.isoformat(),"completed_at":cur.isoformat()}); continue
        end=cur+timedelta(hours=d)
        if is_open and idx==stop-1:
            h.append({"stage":s,"started_at":cur.isoformat(),"completed_at":None})
            st={"investigating":"investigating","physical_count":"physical_count_complete","system_correction":"system_corrected"}.get(s,"investigating"); cur=end
        else: h.append({"stage":s,"started_at":cur.isoformat(),"completed_at":end.isoformat()}); cur=end
    return h,("resolved" if not is_open else st),(cur if not is_open else None)

def generate_variance_events(inv, branches, today):
    bids=list(branches["branch_id"])
    # CHANGE 1: weight discrepancies toward cheap, fast-moving SKUs
    w = inv["velocity_class"].map(VEL_SCORE).values / (inv["unit_cost"].values ** 0.6); w = w/w.sum()
    n_events=int(N_BRANCHES*EVENTS_PER_BRANCH_DAY*HISTORY_DAYS)
    picks = inv.iloc[np.random.choice(np.arange(len(inv)), size=n_events, p=w)].to_dict("records")
    n_open=int(n_events*OPEN_RATIO); rows=[]
    for k,sku in enumerate(picks):
        op = k >= n_events-n_open
        if op: det=today-timedelta(days=random.uniform(0.2,10)); hrs=(today-det).total_seconds()/3600
        else:
            det=today-timedelta(days=random.uniform(2,HISTORY_DAYS)); hrs=float(np.clip(np.random.lognormal(0.5,0.85),0.1,30))*24
            if det+timedelta(hours=hrs)>today: hrs=max(1,(today-det).total_seconds()/3600-1)
        h,st,res=_stage_hist(det,hrs,op)
        # CHANGE 2: small qty, capped +/-1 for expensive items
        base={"A":3,"B":2,"C":1}[sku["velocity_class"]]; mag=max(1,min(4,int(abs(np.random.normal(base,base*0.6)))))
        if sku["unit_cost"]>400: mag=1
        vq=(-1 if random.random()<0.65 else 1)*mag; exp=int(np.random.randint(5,200))
        rows.append({"event_id":str(uuid.uuid4()),"sku_id":sku["sku_id"],"branch_id":random.choice(bids),
            "detection_source":random.choices(SOURCES,weights=SRC_W)[0],"detection_date":det.isoformat(),
            "resolution_date":res.isoformat() if res else None,"expected_qty":exp,"actual_qty":max(0,exp+vq),
            "variance_qty":vq,"variance_cost":round(vq*sku["unit_cost"],2),"stage_history":json.dumps(h),
            "assigned_to":random.choice(HANDLERS),"current_status":st})
    return pd.DataFrame(rows).sort_values("detection_date").reset_index(drop=True)

def main():
    today=datetime.now().replace(microsecond=0)
    inv=generate_inventory(today); br=generate_branches(); ev=generate_variance_events(inv,br,today)
    inv.to_csv(DATA_DIR/"inventory_master.csv",index=False)
    ev.to_csv(DATA_DIR/"variance_events.csv",index=False)
    br.to_csv(DATA_DIR/"branches.csv",index=False)
    tot=ev["variance_cost"].abs().sum()
    print(f"Demo data: {len(inv):,} SKUs | {len(ev):,} events | {len(br)} branches")
    print(f"  absolute variance exposure ${tot:,.0f} | avg ${tot/len(ev):.0f}/event -> {DATA_DIR}")

if __name__=="__main__":
    main()
