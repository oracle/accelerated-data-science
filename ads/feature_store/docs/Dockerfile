FROM python:3.8
USER root
COPY build/html /root/docs/.
WORKDIR /root
EXPOSE 8000
ENTRYPOINT ["python3"]
CMD ["-m", "http.server", "8000"]
