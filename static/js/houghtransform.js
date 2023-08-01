let image_hough = document.querySelector('#imagehoughtransform');
let image_input = document.querySelector('#imageinput');
// let houghtransform_method=document.querySelector("#houghtransform");
let submit=document.querySelector("#submitting_btn");
let selectbox = document.querySelector('#houghtransformselectbox');
let image_data = ""



image_input.addEventListener('change', e => {
    if (e.target.files.length) {
      const reader = new FileReader();
      reader.onload = e => {
        if (e.target.result) {
          let img = document.createElement('img');
          img.id ="imagehoughtransform";
          img.src = e.target.result;
          image_hough.innerHTML = '';
          image_hough.appendChild(img)
          image_data = e.target.result
  
          
        }
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  });




submit.addEventListener('click', e => {
e.preventDefault();
send();
}
)

    
function send(){
        
      let formData = new FormData();

      try {

       if (image_data == "") {
        throw "error : not enought images "
      }
      formData.append('path',image_data);
      formData.append("houghtransformselectbox" ,houghtransformselectbox.value);

    
      console.log("formdata done")
      $.ajax({
        type: 'POST',
        url: '/houghtransform',
        data: formData,
        cache: false,
        contentType: false,
        processData: false,
        async: true,
        success: function (backEndData) {
          var responce = JSON.parse(backEndData)
          let houghtransformoutput = document.getElementById("houghtransformoutput");
          houghtransformoutput.remove();
          houghtransformoutput = document.createElement("div");
          houghtransformoutput.id = "houghtransformoutput";
          houghtransformoutput.innerHTML = responce[1];
          let col2 = document.getElementById("Col2");
          col2.appendChild(houghtransformoutput);
        }

      })
      console.log("ajax done")
    }
     catch (error) {
      console.log("please upload the image")
    } 
  }