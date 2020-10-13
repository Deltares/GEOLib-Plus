# Based on MDCPTData.pas
from ctypes import *
from os.path import splitext
from os import remove
import warnings
from multiprocessing import Pool
from pathlib import Path
import sys

######################## GEF2 - Functions ##############################################
class gef2_dll:

    def __init__(self):

        sourceDLL = r".\\geolib_plus\\resources\\geflib.dll"

        self.hDLL = WinDLL(sourceDLL)
        # Load DLL into memory.

        func = self.hDLL['init_gef']
        func.restype = c_int
        result = func()
        if result == 1:
            return
        else:
            raise Exception("geflib.dll not found")

    def test_gef(self, item):
        func = self.hDLL['test_gef']
        func.restype = c_int
        func.argtype = c_char_p
        return func(item.encode('utf-8'))

    def free_gef(self):
        func = self.hDLL['free_gef']
        func.restype = c_int
        return func()

    def unload_dll(self):
        self.free_gef()
        del self.hDLL

    def read_gef(self, cFile: Path) -> int:
        func = self.hDLL['read_gef']
        func.restype = c_int
        func.argtype = c_char_p
        return func(str(cFile).encode('utf-8'))

    def get_procedurecode_flag(self) -> int:
        func = self.hDLL['get_procedurecode_flag']
        func.restype = c_int
        return func()

    def get_procedurecode_code(self) -> str:
        func = self.hDLL['get_procedurecode_code']
        func.restype = c_char_p
        return func()
    
    def get_procedurecode_release(self) -> int:
        func = self.hDLL['get_procedurecode_release']
        func.restype = c_int
        return func()

    def get_procedurecode_versie(self) -> int:
        func = self.hDLL['get_procedurecode_versie']
        func.restype = c_int
        return func()

    def get_procedurecode_update(self) -> int:
        func = self.hDLL['get_procedurecode_update']
        func.restype = c_int
        return func()

    def get_procedurecode_isoref(self) -> int:
        func = self.hDLL['get_procedurecode_code']
        func.restype = c_char_p
        return func()

    def get_reportcode_flag(self) -> int:
        func = self.hDLL['get_reportcode_flag']
        func.restype = c_int
        return func()

    def get_reportcode_code(self) -> str:
        func = self.hDLL['get_reportcode_code']
        func.restype = c_char_p
        return func()

    def get_reportcode_release(self) -> int:
        func = self.hDLL['get_reportcode_release']
        func.restype = c_int
        return func()

    def get_reportcode_versie(self) -> int:
        func = self.hDLL['get_reportcode_versie']
        func.restype = c_int
        return func()

    def get_reportcode_update(self) -> int:
        func = self.hDLL['get_reportcode_update']
        func.restype = c_int
        return func()

    def get_reportcode_isoref(self) -> int:
        func = self.hDLL['get_reportcode_code']
        func.restype = c_char_p
        return func()

    def get_analysiscode_flag(self) -> int:
        func = self.hDLL['get_analysiscode_flag']
        func.restype = c_int
        return func()

    def get_analysiscode_code(self) -> str:
        func = self.hDLL['get_analysiscode_code']
        func.restype = c_char_p
        return func()
    
    def get_analysiscode_release(self) -> int:
        func = self.hDLL['get_analysiscode_release']
        func.restype = c_int
        return func()

    def get_analysiscode_versie(self) -> int:
        func = self.hDLL['get_analysiscode_versie']
        func.restype = c_int
        return func()

    def get_analysiscode_update(self) -> int:
        func = self.hDLL['get_analysiscode_update']
        func.restype = c_int
        return func()

    def get_analysiscode_isoref(self) -> int:
        func = self.hDLL['get_analysiscode_code']
        func.restype = c_char_p
        return func()
    
    def get_measurementcode_flag(self) -> int:
        func = self.hDLL['get_measurementcode_flag']
        func.restype = c_int
        return func()

    def get_measurementcode_code(self) -> str:
        func = self.hDLL['get_measurementcode_code']
        func.restype = c_char_p
        return func()

    def get_measurementcode_release(self) -> int:
        func = self.hDLL['get_measurementcode_release']
        func.restype = c_int
        return func()

    def get_measurementcode_versie(self) -> int:
        func = self.hDLL['get_measurementcode_versie']
        func.restype = c_int
        return func()

    def get_measurementcode_update(self) -> int:
        func = self.hDLL['get_measurementcode_update']
        func.restype = c_int
        return func()

    def get_measurementcode_isoref(self) -> int:
        func = self.hDLL['get_measurementcode_code']
        func.restype = c_char_p
        return func()

    def get_error_level_all(self) -> int:
        func = self.hDLL['get_error_level_all']
        func.restype = c_int
        return func()

    def get_error_text(self, error_code: int) -> str:
        func = self.hDLL['get_error_text']
        func.restype = c_char_p
        func.argtype = c_int
        return func(error_code).decode('utf-8')

    def write_error_log(self, cFile: str) -> int:
        func = self.hDLL['write_error_log']
        func.restype = c_int
        func.argtype = c_char_p
        return func(cFile.encode('utf-8'))

######################## D Series - GeoLib Function ##############################################

def GetCode(ACode: str) -> str:

    CsGEFBOREReport: str = 'GEF-BORE-REPORT'
    CsGEFCPTReport: str = 'GEF-CPT-REPORT'
    CsShortGEFCPTReport: str = 'CPT-REPORT'
    CsGEFZSTEENINVOER: str = 'GEF-ZSTEEN-INVOER'

    ACode = ACode.decode("utf-8").strip(' ').upper()

    if (ACode == CsGEFCPTReport) or (ACode == CsShortGEFCPTReport):
        return "gefCPTReport"

    if 'CPT-A' in ACode:
        return "gefCPTAnalysis"

    if 'CPT-M' in ACode:
        return "gefCPTMeasurement"

    if ACode == CsGEFBOREReport:
        return "gefBoringReport"

    if 'BoreA' in ACode:
        return "gefBoringAnalysis"

    if 'BoreM' in ACode:
        return "gefBoringMeasurement"

    if ACode == CsGEFZSTEENINVOER:
        return "gefWave"

    return None


def GetGefType(hDLL: gef2_dll, AFileName) -> str:
    Result = None  # gef unknown
    hDLL.read_gef(AFileName)
    # Try for REPORT first( for CPTs, this can be in reportcode or in procedurecode)
    flag = hDLL.get_procedurecode_flag()
    if flag == 1:
        return GetCode(hDLL.get_procedurecode_code())

    if Result is None:
        flag = hDLL.get_reportcode_flag()
        if flag == 1:
            return GetCode(hDLL.get_reportcode_code())

    # when GEF is not recognized as a kind of report, try for Analysis
    if Result is None:
        flag = hDLL.get_analysiscode_flag()
        if flag == 1:
            return GetCode(hDLL.get_analysiscode_code())

    # when GEF is still not recognized, try for Measurement
    if Result is None:
        flag = hDLL.get_measurementcode_flag()
        if flag == 1:
            return GetCode(hDLL.get_measurementcode_code())

    # when GEF is still not recognized, try for Database BORE Measurement file. To be recognized by GEF-BORE as code
    if Result is None:
        flag = hDLL.get_procedurecode_flag()
        if flag == 1:
            LCode = hDLL.get_procedurecode_code().upper()
            if 'GEF-BORE' in LCode:
                return "gefBoringMeasurement"
    else:
        return None


def DoesGEFContainSeriousErrors(hDLL: gef2_dll) -> (bool, bool, int):
    hDLL.test_gef('Header')

    LLog = False
    LErrorLevel = hDLL.get_error_level_all()
    LError = LErrorLevel in [1, 2, 11, 12]

    if not LError:
        # Only continue check when nothing serious so far.
        hDLL.test_gef('Data')
        LErrorLevel = hDLL.get_error_level_all()
        LError = LErrorLevel in [1, 2, 11, 12]

    if (LErrorLevel > -1) and (LErrorLevel not in [5, 6, 15, 16]):
        LLog = True

    return LError, LLog, LErrorLevel


def ValidateCPTFromGEF(AFileName: str, ALogging: bool = True) -> int:

    if ALogging:
        # file name for err \ delete existing if necessary
        pre, ext = splitext(AFileName)
        LErrFileName = pre + '.err'
        try:
            # remove file if present
            remove(LErrFileName)
        except:
            pass
    try:
        hDLL = gef2_dll()
        gef_type = GetGefType(hDLL, AFileName)

        if gef_type in ['gefCPTReport', 'gefCPTMeasurement', 'gefCPTAnalysis']:  # JN: I think we might only want gefCPTReport
            LError, LLog, LErrorLevel = DoesGEFContainSeriousErrors(hDLL)

            if LLog and ALogging:
                hDLL.write_error_log(LErrFileName)

            if LError:
                if ALogging:
                    raise Exception('Validation Error: Severe GEF Error: Code ' + str(LErrorLevel) +
                                    ' - See Log File: ' + LErrFileName + ')')
                else:
                    raise Exception('Validation Error: Severe GEF Error: Code ' + str(LErrorLevel) +
                                    ' - Check that the file conforms to GEF CPT Standard')
            elif LLog:
                if ALogging:
                    warnings.warn('Validation Warning: This file has raised a  '
                                  'GEF warning (Code ' + str(LErrorLevel) + ' - See Log File: ' + LErrFileName + ')')
                else:
                    warnings.warn('Validation Warning: This file has raised the '
                                  'GEF warning(Code ' + str(LErrorLevel) + '). Check that the file conforms to GEF CPT Standard')
        else:
            raise Exception("GEF File is not of type CPT Type:", gef_type)
    finally:
        hDLL.unload_dll()
        del hDLL
    return 0

def ExecuteGEFValidation(AFileName: str, ALogging: bool = True) -> int:
    pool = Pool(processes=1)
    result = pool.apply_async(ValidateCPTFromGEF, args=(AFileName, ALogging))
    result.get()
    pool.close()
    pool.join()











