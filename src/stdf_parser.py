"""
STDF V4 Binary File Parser
============================

Standard Test Data Format (STDF) parser for semiconductor ATE data.
Supports Advantest V93000, Teradyne J750/UltraFLEX output files.

STDF Record Structure:
┌──────────┬──────────┬──────────┬──────────┐
│ REC_LEN  │ REC_TYP  │ REC_SUB  │   DATA   │
│ (2 bytes)│ (1 byte) │ (1 byte) │(variable)│
└──────────┴──────────┴──────────┴──────────┘

Author: Mst Arefin Aktar
Date: 2026
"""

import struct
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import warnings


# ============================================================
# STDF Record Definitions
# ============================================================

RECORD_MAP = {
    (0, 10): 'FAR', (0, 20): 'ATR',
    (1, 10): 'MIR', (1, 20): 'MRR',
    (1, 30): 'PCR', (1, 40): 'HBR',
    (1, 50): 'SBR', (1, 60): 'PMR',
    (1, 62): 'PGR', (1, 63): 'PLR',
    (1, 70): 'RDR', (1, 80): 'SDR',
    (2, 10): 'WIR', (2, 20): 'WRR', (2, 30): 'WCR',
    (5, 10): 'PIR', (5, 20): 'PRR',
    (10, 30): 'TSR',
    (15, 10): 'PTR', (15, 15): 'MPR', (15, 20): 'FTR',
    (50, 10): 'GDR', (50, 30): 'DTR',
}


@dataclass
class MIR:
    """Master Information Record"""
    setup_t: int = 0
    start_t: int = 0
    stat_num: int = 0
    mode_cod: str = ""
    rtst_cod: str = ""
    prot_cod: str = ""
    burn_tim: int = 0
    cmod_cod: str = ""
    lot_id: str = ""
    part_typ: str = ""
    node_nam: str = ""
    tstr_typ: str = ""
    job_nam: str = ""
    job_rev: str = ""
    sblot_id: str = ""
    oper_nam: str = ""
    exec_typ: str = ""
    exec_ver: str = ""
    test_cod: str = ""
    tst_temp: str = ""
    user_txt: str = ""
    aux_file: str = ""
    pkg_typ: str = ""
    famly_id: str = ""
    date_cod: str = ""
    facil_id: str = ""
    floor_id: str = ""
    proc_id: str = ""


@dataclass
class PTR:
    """Parametric Test Record"""
    test_num: int = 0
    head_num: int = 0
    site_num: int = 0
    test_flg: int = 0
    parm_flg: int = 0
    result: float = 0.0
    test_txt: str = ""
    alarm_id: str = ""
    opt_flag: int = 0
    res_scal: int = 0
    llm_scal: int = 0
    hlm_scal: int = 0
    lo_limit: Optional[float] = None
    hi_limit: Optional[float] = None
    units: str = ""
    c_resfmt: str = ""
    c_llmfmt: str = ""
    c_hlmfmt: str = ""
    lo_spec: Optional[float] = None
    hi_spec: Optional[float] = None


@dataclass
class PRR:
    """Part Results Record"""
    head_num: int = 0
    site_num: int = 0
    part_flg: int = 0
    num_test: int = 0
    hard_bin: int = 0
    soft_bin: int = 0
    x_coord: int = -32768
    y_coord: int = -32768
    test_t: int = 0
    part_id: str = ""
    part_txt: str = ""


@dataclass
class FTR:
    """Functional Test Record"""
    test_num: int = 0
    head_num: int = 0
    site_num: int = 0
    test_flg: int = 0
    opt_flag: int = 0
    cycl_cnt: int = 0
    rel_vadr: int = 0
    rept_cnt: int = 0
    num_fail: int = 0
    xfail_ad: int = 0
    yfail_ad: int = 0
    vect_off: int = 0
    pat_name: str = ""
    test_txt: str = ""
    alarm_id: str = ""


@dataclass
class TSR:
    """Test Synopsis Record"""
    head_num: int = 0
    site_num: int = 0
    test_typ: str = ""
    test_num: int = 0
    exec_cnt: int = 0
    fail_cnt: int = 0
    alrm_cnt: int = 0
    test_nam: str = ""
    seq_name: str = ""
    test_lbl: str = ""
    test_tim: float = 0.0
    test_min: float = 0.0
    test_max: float = 0.0
    tst_sums: float = 0.0
    tst_sqrs: float = 0.0


# ============================================================
# Main Parser Class
# ============================================================

class STDFV4Parser:
    """
    Full STDF V4 Parser for Advantest V93000 / Teradyne

    Usage:
        parser = STDFV4Parser("path/to/file.stdf")
        parser.parse()
        parser.print_summary()

        df = parser.get_ptr_dataframe(site=0)
        parser.to_csv("output.csv")
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.endian = '<'
        self.stdf_ver = 4

        self.far_data = None
        self.mir: Optional[MIR] = None
        self.ptr_records: List[PTR] = []
        self.prr_records: List[PRR] = []
        self.ftr_records: List[FTR] = []
        self.tsr_records: List[TSR] = []
        self.record_counts: Dict[str, int] = {}

        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self._file_size = self.filepath.stat().st_size
        print(f"File: {self.filepath.name}")
        print(f"Size: {self._file_size:,} bytes ({self._file_size / 1024:.1f} KB)")

    # ---- Low-level readers ----

    def _read_u1(self, data: bytes, offset: int) -> Tuple[int, int]:
        if offset >= len(data):
            return 0, offset
        return struct.unpack('B', data[offset:offset + 1])[0], offset + 1

    def _read_u2(self, data: bytes, offset: int) -> Tuple[int, int]:
        if offset + 2 > len(data):
            return 0, offset
        return struct.unpack(f'{self.endian}H', data[offset:offset + 2])[0], offset + 2

    def _read_u4(self, data: bytes, offset: int) -> Tuple[int, int]:
        if offset + 4 > len(data):
            return 0, offset
        return struct.unpack(f'{self.endian}I', data[offset:offset + 4])[0], offset + 4

    def _read_i1(self, data: bytes, offset: int) -> Tuple[int, int]:
        if offset >= len(data):
            return 0, offset
        return struct.unpack('b', data[offset:offset + 1])[0], offset + 1

    def _read_i2(self, data: bytes, offset: int) -> Tuple[int, int]:
        if offset + 2 > len(data):
            return 0, offset
        return struct.unpack(f'{self.endian}h', data[offset:offset + 2])[0], offset + 2

    def _read_r4(self, data: bytes, offset: int) -> Tuple[float, int]:
        if offset + 4 > len(data):
            return 0.0, offset
        return struct.unpack(f'{self.endian}f', data[offset:offset + 4])[0], offset + 4

    def _read_cn(self, data: bytes, offset: int) -> Tuple[str, int]:
        if offset >= len(data):
            return "", offset
        str_len, offset = self._read_u1(data, offset)
        if offset + str_len > len(data):
            return "", offset
        try:
            string = data[offset:offset + str_len].decode('ascii', errors='replace')
        except Exception:
            string = ""
        return string, offset + str_len

    def _read_c1(self, data: bytes, offset: int) -> Tuple[str, int]:
        if offset >= len(data):
            return " ", offset
        try:
            char = data[offset:offset + 1].decode('ascii', errors='replace')
        except Exception:
            char = " "
        return char, offset + 1

    # ---- Record Parsers ----

    def _parse_far(self, data: bytes):
        offset = 0
        cpu_type, offset = self._read_u1(data, offset)
        stdf_ver, offset = self._read_u1(data, offset)

        if cpu_type == 2:
            self.endian = '<'
        elif cpu_type == 1:
            self.endian = '>'
        else:
            self.endian = '<'

        self.stdf_ver = stdf_ver
        endian_str = 'Little-Endian' if self.endian == '<' else 'Big-Endian'
        print(f"STDF V{stdf_ver} | CPU: {cpu_type} ({endian_str})")

    def _parse_mir(self, data: bytes):
        mir = MIR()
        offset = 0

        mir.setup_t, offset = self._read_u4(data, offset)
        mir.start_t, offset = self._read_u4(data, offset)
        mir.stat_num, offset = self._read_u1(data, offset)
        mir.mode_cod, offset = self._read_c1(data, offset)
        mir.rtst_cod, offset = self._read_c1(data, offset)
        mir.prot_cod, offset = self._read_c1(data, offset)
        mir.burn_tim, offset = self._read_u2(data, offset)
        mir.cmod_cod, offset = self._read_c1(data, offset)
        mir.lot_id, offset = self._read_cn(data, offset)
        mir.part_typ, offset = self._read_cn(data, offset)
        mir.node_nam, offset = self._read_cn(data, offset)
        mir.tstr_typ, offset = self._read_cn(data, offset)
        mir.job_nam, offset = self._read_cn(data, offset)
        mir.job_rev, offset = self._read_cn(data, offset)
        mir.sblot_id, offset = self._read_cn(data, offset)
        mir.oper_nam, offset = self._read_cn(data, offset)
        mir.exec_typ, offset = self._read_cn(data, offset)
        mir.exec_ver, offset = self._read_cn(data, offset)

        self.mir = mir

        start_time = datetime.fromtimestamp(mir.start_t).strftime('%Y-%m-%d %H:%M:%S') if mir.start_t else "N/A"
        print(f"\nLot: {mir.lot_id} | Part: {mir.part_typ} | Node: {mir.node_nam}")
        print(f"Tester: {mir.tstr_typ} | Job: {mir.job_nam}")
        print(f"Software: {mir.exec_typ} {mir.exec_ver}")
        print(f"Start: {start_time}")

    def _parse_ptr(self, data: bytes) -> Optional[PTR]:
        ptr = PTR()
        offset = 0
        try:
            ptr.test_num, offset = self._read_u4(data, offset)
            ptr.head_num, offset = self._read_u1(data, offset)
            ptr.site_num, offset = self._read_u1(data, offset)
            ptr.test_flg, offset = self._read_u1(data, offset)
            ptr.parm_flg, offset = self._read_u1(data, offset)
            ptr.result, offset = self._read_r4(data, offset)
            ptr.test_txt, offset = self._read_cn(data, offset)
            ptr.alarm_id, offset = self._read_cn(data, offset)

            if offset < len(data):
                ptr.opt_flag, offset = self._read_u1(data, offset)
            if offset < len(data):
                ptr.res_scal, offset = self._read_i1(data, offset)
            if offset < len(data):
                ptr.llm_scal, offset = self._read_i1(data, offset)
            if offset < len(data):
                ptr.hlm_scal, offset = self._read_i1(data, offset)
            if offset + 4 <= len(data):
                ptr.lo_limit, offset = self._read_r4(data, offset)
            if offset + 4 <= len(data):
                ptr.hi_limit, offset = self._read_r4(data, offset)
            if offset < len(data):
                ptr.units, offset = self._read_cn(data, offset)
            if offset < len(data):
                ptr.c_resfmt, offset = self._read_cn(data, offset)
            if offset < len(data):
                ptr.c_llmfmt, offset = self._read_cn(data, offset)
            if offset < len(data):
                ptr.c_hlmfmt, offset = self._read_cn(data, offset)
            if offset + 4 <= len(data):
                ptr.lo_spec, offset = self._read_r4(data, offset)
            if offset + 4 <= len(data):
                ptr.hi_spec, offset = self._read_r4(data, offset)

            self.ptr_records.append(ptr)
            return ptr
        except Exception as ex:
            warnings.warn(f"PTR parse error: {ex}")
            return None

    def _parse_ftr(self, data: bytes) -> Optional[FTR]:
        ftr = FTR()
        offset = 0
        try:
            ftr.test_num, offset = self._read_u4(data, offset)
            ftr.head_num, offset = self._read_u1(data, offset)
            ftr.site_num, offset = self._read_u1(data, offset)
            ftr.test_flg, offset = self._read_u1(data, offset)
            self.ftr_records.append(ftr)
            return ftr
        except Exception as ex:
            warnings.warn(f"FTR parse error: {ex}")
            return None

    def _parse_prr(self, data: bytes) -> Optional[PRR]:
        prr = PRR()
        offset = 0
        try:
            prr.head_num, offset = self._read_u1(data, offset)
            prr.site_num, offset = self._read_u1(data, offset)
            prr.part_flg, offset = self._read_u1(data, offset)
            prr.num_test, offset = self._read_u2(data, offset)
            prr.hard_bin, offset = self._read_u2(data, offset)
            prr.soft_bin, offset = self._read_u2(data, offset)
            prr.x_coord, offset = self._read_i2(data, offset)
            prr.y_coord, offset = self._read_i2(data, offset)
            prr.test_t, offset = self._read_u4(data, offset)
            prr.part_id, offset = self._read_cn(data, offset)
            prr.part_txt, offset = self._read_cn(data, offset)
            self.prr_records.append(prr)
            return prr
        except Exception as ex:
            warnings.warn(f"PRR parse error: {ex}")
            return None

    def _parse_tsr(self, data: bytes) -> Optional[TSR]:
        tsr = TSR()
        offset = 0
        try:
            tsr.head_num, offset = self._read_u1(data, offset)
            tsr.site_num, offset = self._read_u1(data, offset)
            tsr.test_typ, offset = self._read_c1(data, offset)
            tsr.test_num, offset = self._read_u4(data, offset)
            tsr.exec_cnt, offset = self._read_u4(data, offset)
            tsr.fail_cnt, offset = self._read_u4(data, offset)
            tsr.alrm_cnt, offset = self._read_u4(data, offset)
            tsr.test_nam, offset = self._read_cn(data, offset)
            tsr.seq_name, offset = self._read_cn(data, offset)
            tsr.test_lbl, offset = self._read_cn(data, offset)
            if offset + 4 <= len(data):
                tsr.test_tim, offset = self._read_r4(data, offset)
            if offset + 4 <= len(data):
                tsr.test_min, offset = self._read_r4(data, offset)
            if offset + 4 <= len(data):
                tsr.test_max, offset = self._read_r4(data, offset)
            if offset + 4 <= len(data):
                tsr.tst_sums, offset = self._read_r4(data, offset)
            if offset + 4 <= len(data):
                tsr.tst_sqrs, offset = self._read_r4(data, offset)
            self.tsr_records.append(tsr)
            return tsr
        except Exception as ex:
            warnings.warn(f"TSR parse error: {ex}")
            return None

    # ---- Main Parse ----

    def parse(self):
        print(f"\n{'=' * 50}")
        print(f"PARSING: {self.filepath.name}")
        print(f"{'=' * 50}")

        total_records = 0

        with open(self.filepath, 'rb') as f:
            while True:
                header_bytes = f.read(4)
                if len(header_bytes) < 4:
                    break

                rec_len = struct.unpack(f'{self.endian}H', header_bytes[0:2])[0]
                rec_typ = struct.unpack('B', header_bytes[2:3])[0]
                rec_sub = struct.unpack('B', header_bytes[3:4])[0]

                data = f.read(rec_len)
                if len(data) < rec_len:
                    break

                rec_name = RECORD_MAP.get((rec_typ, rec_sub), f'UNK({rec_typ},{rec_sub})')
                self.record_counts[rec_name] = self.record_counts.get(rec_name, 0) + 1

                if rec_typ == 0 and rec_sub == 10:
                    self._parse_far(data)
                elif rec_typ == 1 and rec_sub == 10:
                    self._parse_mir(data)
                elif rec_typ == 15 and rec_sub == 10:
                    self._parse_ptr(data)
                elif rec_typ == 15 and rec_sub == 20:
                    self._parse_ftr(data)
                elif rec_typ == 5 and rec_sub == 20:
                    self._parse_prr(data)
                elif rec_typ == 10 and rec_sub == 30:
                    self._parse_tsr(data)

                total_records += 1

        print(f"\nTotal records: {total_records:,}")
        print(f"PTR: {len(self.ptr_records):,} | FTR: {len(self.ftr_records):,} | PRR: {len(self.prr_records):,}")

    # ---- Data Export ----

    def get_ptr_dataframe(self, site: Optional[int] = None) -> pd.DataFrame:
        if not self.ptr_records:
            raise ValueError("No PTR records. Call parse() first.")

        rows = []
        for ptr in self.ptr_records:
            if site is not None and ptr.site_num != site:
                continue

            passed = not bool(ptr.test_flg & 0x80)
            executed = not bool(ptr.test_flg & 0x40)

            rows.append({
                'test_num': ptr.test_num,
                'test_name': ptr.test_txt,
                'site': ptr.site_num,
                'head': ptr.head_num,
                'result': ptr.result,
                'lo_limit': ptr.lo_limit,
                'hi_limit': ptr.hi_limit,
                'units': ptr.units,
                'pass_fail': 'PASS' if passed else 'FAIL',
                'executed': executed,
                'test_flag': ptr.test_flg,
                'alarm': ptr.alarm_id,
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            print(f"DataFrame: {len(df)} rows, {df['test_name'].nunique()} unique tests")
        return df

    def get_prr_dataframe(self) -> pd.DataFrame:
        rows = []
        for prr in self.prr_records:
            passed = not bool(prr.part_flg & 0x08)
            rows.append({
                'site': prr.site_num,
                'hard_bin': prr.hard_bin,
                'soft_bin': prr.soft_bin,
                'x_coord': prr.x_coord,
                'y_coord': prr.y_coord,
                'num_test': prr.num_test,
                'test_time_ms': prr.test_t,
                'part_id': prr.part_id,
                'pass_fail': 'PASS' if passed else 'FAIL',
            })
        return pd.DataFrame(rows)

    def to_csv(self, output_path: str, site: Optional[int] = None):
        df = self.get_ptr_dataframe(site=site)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} records to {output_path}")

    def get_test_summary(self, site: int = 0) -> pd.DataFrame:
        df = self.get_ptr_dataframe(site=site)
        if df.empty:
            return pd.DataFrame()

        summary = df.groupby(['test_num', 'test_name', 'units']).agg(
            count=('result', 'count'),
            mean=('result', 'mean'),
            std=('result', 'std'),
            min_val=('result', 'min'),
            max_val=('result', 'max'),
            lo_limit=('lo_limit', 'first'),
            hi_limit=('hi_limit', 'first'),
            pass_count=('pass_fail', lambda x: (x == 'PASS').sum()),
            fail_count=('pass_fail', lambda x: (x == 'FAIL').sum()),
        ).reset_index()

        summary['yield_pct'] = (summary['pass_count'] / summary['count'] * 100).round(2)
        return summary

    def print_summary(self):
        print(f"\n{'=' * 60}")
        print(f"  STDF FILE SUMMARY")
        print(f"{'=' * 60}")

        if self.mir:
            print(f"  Lot:       {self.mir.lot_id}")
            print(f"  Part:      {self.mir.part_typ}")
            print(f"  Tester:    {self.mir.tstr_typ}")
            print(f"  Program:   {self.mir.job_nam}")
            print(f"  Node:      {self.mir.node_nam}")

        print(f"\n  Records:")
        for rec, count in sorted(self.record_counts.items()):
            print(f"    {rec:8s}: {count:,}")

        if self.ptr_records:
            df = self.get_ptr_dataframe()
            sites = sorted(df['site'].unique())
            tests = df['test_name'].nunique()
            total = len(df)
            pass_count = (df['pass_fail'] == 'PASS').sum()

            print(f"\n  Test Data:")
            print(f"    Sites:        {sites}")
            print(f"    Unique tests: {tests}")
            print(f"    Total PTR:    {total:,}")
            print(f"    Pass:         {pass_count:,} ({pass_count / total * 100:.1f}%)")
            print(f"    Fail:         {total - pass_count:,}")

        print(f"{'=' * 60}")
