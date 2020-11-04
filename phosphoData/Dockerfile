from amancevice/pandas

RUN pip install cptac
COPY phosphoPDCdata.py /bin/
RUN chmod 777 /bin/phosphoPDCdata.py
RUN python /bin/phosphoPDCdata.py --getData

VOLUME ['/tmp']
