document.addEventListener('DOMContentLoaded', function() {
  var gallery = document.querySelector('#results-carousel');
  if (!gallery || typeof bulmaCarousel === 'undefined') {
    return;
  }

  var carousel = bulmaCarousel.attach(gallery, {
    slidesToScroll: 1,
    slidesToShow: 1,
    // The library's default breakpoints show multiple slides on smaller screens.
    breakpoints: [],
    loop: true,
    // Infinite mode clones video elements, including their autoplay attributes.
    infinite: false,
    autoplay: false
  })[0];

  function playSlide(index) {
    carousel.slides.forEach(function(slide) {
      var video = slide.querySelector('video');
      var isActive = Number(slide.dataset.sliderIndex) === Number(index);
      slide.setAttribute('aria-hidden', String(!isActive));
      video.tabIndex = isActive ? 0 : -1;

      if (isActive) {
        // Keep native controls usable when the browser blocks autoplay.
        video.play().catch(function() {});
      } else {
        video.pause();
      }
    });
  }

  gallery.querySelectorAll('video').forEach(function(video) {
    // Scrubbing a video must not also drag the carousel.
    ['mousedown', 'touchstart'].forEach(function(eventName) {
      video.addEventListener(eventName, function(event) {
        event.stopPropagation();
      });
    });
  });

  carousel.on('before:show', function(state) {
    // Pagination also supplies a string index; keep subsequent navigation numeric.
    state.next = Number(state.next);
    // The library updates state.index only after the slide transition finishes.
    playSlide(state.next);
  });
  playSlide(carousel.state.index);

  var galleryWidth = gallery.clientWidth;
  new ResizeObserver(function() {
    if (gallery.clientWidth === galleryWidth) {
      return;
    }
    galleryWidth = gallery.clientWidth;
    // Bulma only recalculates widths when the number of visible slides changes.
    carousel._setDimensions();
    carousel.transitioner.init().apply(true);
    carousel.onShow();
  }).observe(gallery);
});

document.addEventListener('DOMContentLoaded', () => {

  bulmaCarousel.attach('#results-carousel', {

    slidesToScroll: 1,

    slidesToShow: 1,

    loop: true,

    autoplay: false,

    pagination: true,

    navigation: true

  });

});