from botocore.exceptions import NoCredentialsError, ClientError
from fastapi import HTTPException

def allowed_file(filename, state, consts):

    ext = filename.rsplit(".", 1)[-1].lower()
    return "." in filename and ext in consts.ALLOWED_EXTENSIONS

def upload_to_r2(local_path, r2_key, state, consts):
    
    """
    uploads a local file to the r2 bucket.
    returns the url or raises an HTTPException.
    """

    try:
        state.s3.upload_file(local_path, consts.R2_BUCKET, r2_key) # upload file
        public_url = f"{consts.R2_PUBLIC_URL}/{r2_key}"
        return public_url
    
    except NoCredentialsError:
        msg = "invalid r2 credentials, check r2_key and r2_secret"
        print(msg)
        raise HTTPException(status_code=500, detail=msg)
    
    except ClientError as e:
        code = e.response['Error']['Code']
        message = e.response['Error']['Message']
        msg = f"r2 upload failed: {code} - {message}"
        print(msg)
        raise HTTPException(status_code=500, detail=msg)