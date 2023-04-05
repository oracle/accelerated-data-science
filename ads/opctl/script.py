import json
import os
import sys
import tempfile
import uuid

from ads.model.generic_model import GenericModel


def verify(ocid, data, compartment_id, project_id):
    with tempfile.TemporaryDirectory() as td:
        model = GenericModel.from_model_catalog(ocid,
                                                artifact_dir=os.path.join(td, str(uuid.uuid4())),
                                                force_overwrite=True, compartment_id=compartment_id, project_id=project_id)
        data = json.loads(data)
        print(model.verify(data, auto_serialize_data=False))


def main():
    args = sys.argv[1:]
    verify(ocid = args[0], data=args[1], compartment_id=args[2], project_id=args[3])
    return 0


if __name__ == "__main__":
    main()
