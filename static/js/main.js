$(document).ready(function() {
    if (!window.console) window.console = {};
    if (!window.console.log) window.console.log = function() {};
    var submit = $('#mybutton');
    var form = $('#messageform');
    var results = $('#results');
    var result = $('#result');
    var review = $('#review');
    var text = $('#mytext');
    results.hide()
    form.on('submit',function(e) {
        e.preventDefault(); // prevent default form submit

        $.ajax({
          url: '/', // form action url
          type: 'GET', // form submit method get/post
        dataType: 'text', // request type html/json/xml
        data: form.serialize(), // serialize form data 
        beforeSend: function() {
            submit.html('Predicting....'); // change submit button text
            results.hide();
        },
        success: function(data) {
        result.html(data)
        review.html(text.val());
        results.fadeIn();
        submit.html('Predict!'); // reset submit button text
      },
      error: function(e) {
        console.log(e)
      }
    });
    });
});

