//document.addEventListener('DOMContentLoaded', function() {
//    const imageInput = document.getElementById('image-input');
//    const imagePreview = document.getElementById('image-preview');
//    const imagePreviewContainer = document.getElementById('image-preview-container');
//
//    imageInput.addEventListener('change', function(event) {
//        if (event.target.files.length > 0) {
//            const file = event.target.files[0];
//            const imageUrl = URL.createObjectURL(file);
//
//            imagePreview.src = imageUrl;
//            imagePreviewContainer.style.display = 'block';
//        } else {
//            imagePreview.src = '';
//            imagePreviewContainer.style.display = 'none';
//        }
//    });
//});



        // 이미지 미리보기를 처리하는 JavaScript 코드
//    document.getElementById('imageInput').addEventListener('change', function(event) {
//        var imagePreview = document.getElementById('imagePreview');
//        imagePreview.innerHTML = '';
//        var file = event.target.files[0];
//        var image = document.createElement('img');
//        image.src = URL.createObjectURL(file);
//        image.style.maxWidth = '300px'; // 미리보기 이미지의 최대 너비 설정
//        image.style.maxHeight = '300px'; // 미리보기 이미지의 최대 높이 설정
//        imagePreview.appendChild(image);
//    });
